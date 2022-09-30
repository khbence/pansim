#!python
import re
import threading
import subprocess
import time
import os
import json
from tokenize import String
import util_funs as uf
import pandas as pd

tmpdirPath = '/home/reguly/pansim/tmpdirs'
panSimPath = '/home/reguly/pansim/'
submitScriptPath = tmpdirPath+'/submit_gpu.sh'
binaryPath = panSimPath+'/build_v100/panSim'

def is_number(s):
    if isinstance(s,dict):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

class uniqueIDProvider:
    lock = threading.Lock()
    counter = 0
    @staticmethod
    def getUniqueID() -> int:
        with uniqueIDProvider.lock:
            uniqueIDProvider.counter = uniqueIDProvider.counter+1
            return uniqueIDProvider.counter


class Instance:
    def __init__(self, nruns, globalConfig, parameters, manager) -> None:
        self.completed = False
        self.globalConfig = globalConfig
        self.parameters = parameters
        self.uniqueID = uniqueIDProvider.getUniqueID()
        self.slurmID = -1
        self.workdir = tmpdirPath+'/'+str(self.uniqueID)
        self.manager = manager
        self.nruns = nruns
        self.results = {}

    def prepare(self) -> None:
        
        #create tmpdir
        if os.path.exists(self.workdir):
            print('Warning, path '+self.workdir+' already exists')
        else:
            os.mkdir(self.workdir)
        
        for key in self.parameters:
            idx = next((i for i, x in enumerate(self.globalConfig) if x==key), None)
            if idx:
                if len(self.globalConfig)>idx+2:
                    self.globalConfig = self.globalConfig[:idx]+self.globalConfig[idx+2:]
                else:
                    self.globalConfig = self.globalConfig[:idx]


        #copy in/create any files/params in "parameters"
        self.localConfig = self.manager.convertParams(self.parameters, self.workdir)
        #assemble command line - use defaults for files/params in "globalConfig"
        # globalC = ' '.join(self.globalConfig)
        # localC = ' '.join(self.localConfig)
        # self.cmd = binaryPath + ' ' + globalC + ' ' + localC
        self.cmd = [binaryPath] + self.globalConfig + self.localConfig


    def run(self) -> None:
        self.prepare()
        #submit job
        fullcmd = ['sbatch',submitScriptPath,str(self.nruns),self.workdir]+self.cmd
        with open(self.workdir+f'/runProperties.txt', 'w') as outfile:
            outfile.write(' '.join(fullcmd))
        print(' '.join(fullcmd))
        a = subprocess.run(fullcmd,capture_output=True)
        if len(a.stderr)>0 or not ('Submitted batch job' in str(a.stdout)):
            print(f'Error submitting job {self.uniqueID}')
            return
        self.slurmID = int(a.stdout[len('Submitted batch job '):])
        self.poll()

    def poll(self) -> None:
        a = subprocess.run(['squeue'],capture_output=True)
        while not (re.search(r'\b'+str(self.slurmID)+r'\b',str(a.stdout)) is None):
            time.sleep(5)
            a = subprocess.run(['squeue'],capture_output=True)
        #parse results
        self.completed = True
        for i in range(0,self.nruns):
            try:
                self.results[i] = pd.DataFrame(uf.std_txt_reader(self.workdir+f'/result_{i+1}.txt'))
            except:
                print(f'Error, {self.workdir} run {i} crashed')
        result = int(0)
        for res in self.results:
            if isinstance(result, int):
                result = self.results[res]
            else:
                result = pd.concat((result, self.results[res]))
        if isinstance(result, int):
            print(f'Error, {self.workdir} batch had no valid runs')
            self.result_avg = pd.DataFrame()
            self.result_std = pd.DataFrame()
        else:
            by_row_index = result.groupby(result.index)
            self.result_avg = by_row_index.mean()
            self.result_std = by_row_index.std()

class Manager:
    def __init__(self, globalConfig) -> None:
        f = open('defaultargs.json')
        self.argsDefaults = json.load(f)
        f.close()
        if type(globalConfig) is dict:
            self.globalConfig = globalConfig
        else:
            if os.path.exists(globalConfig):
                f.open(globalConfig)
                self.globalConfig = json.load(f)
                f.close()
        self.globalArgs = self.convertParams(self.globalConfig, tmpdirPath)

    def convertParams(self, paramlist, directory):
        argstr = []
        #check for consistency, and if some args are dicts, write them to json, concat arg to str
        for key in paramlist:
            if not key in self.argsDefaults:
                print(f'argument {key} not recognized, dropping...')
            else:
                is_a = isinstance(paramlist[key], (int, float)) or is_number(paramlist[key])
                is_b = isinstance(self.argsDefaults[key], (int, float)) or is_number(self.argsDefaults[key])
                if is_a and is_b:
                    argstr = argstr + [key] + [str(paramlist[key])]
                elif (not is_a) and (not is_b):
                    #if the value is a dict, write to json
                    if type(paramlist[key]) is dict:
                        #if the default parameter string is not a file (or the file doesn't exist, but it should)
                        if not os.path.exists(panSimPath+'/'+self.argsDefaults[key]):
                            print(f'got a dictionary as an argument to {key}, but that argument does not take a file as an argument')
                        #if json, write json
                        if 'json' in self.argsDefaults[key]:
                            with open(directory+f'/argsFor{key}.json', 'w') as outfile:
                                json.dump(paramlist[key], outfile)
                        else: #otherwise just write as string
                            with open(directory+f'/argsFor{key}.json', 'w') as outfile:
                                outfile.write(paramlist[key])
                        argstr = argstr + [key] + [directory+f'/argsFor{key}.json']
                    else: #otherwise just string arg
                        argstr = argstr + [key] + [paramlist[key]]
                else:
                    print(f'mistmatch for key {key} between argument types: {paramlist[key]} ({type(paramlist[key])}) and {self.argsDefaults[key]} ({type(self.argsDefaults[key])}), dropping key')
        return argstr

    def createInstance(self, nruns, parameters) -> Instance:
        instanceArgs = self.convertParams(parameters, tmpdirPath)
        instance = Instance(nruns, self.globalArgs, parameters, self)
        return instance

    def runBatch(self, nruns, parameters):
        batchsize = len(parameters)
        threads = []
        instances = []
        print(f'Starting batch of {batchsize}')
        for i in range(0,batchsize):
            instances = instances + [self.createInstance(nruns,parameters[i])]
            threads = threads + [threading.Thread(target=instances[i].run)]
            threads[i].start()
        for i in range(0,batchsize):
            threads[i].join()
        return instances


def main():
    manager = Manager({'-w':2, '-r': ' ',
                        '--infectiousnessMultiplier': '1.05,1.94,2.6,3.2,4.16',
                        '--diseaseProgressionScaling': '0.8,0.98,1.1,0.7,0.7',
                        '--diseaseProgressionDeathScaling': '1.0,1.15,1.25,0.6,0.6',
                        '--acquiredMultiplier': '0.9,0.22,0.8,0.22,0.85,0.15,0.30,0.5,0.30,0.5',
                        '--immunizationEfficiencyInfection': '0.52,0.96,0.99,0.2,0.82,0.95,0.09,0.71,0.91,0.01,0.3,0.54,0.01,0.3,0.54',
                        '--immunizationEfficiencyProgression': '1.0,1.0,1.0,0.4,0.22,0.1,0.4,0.22,0.1,0.67,0.6,0.35,0.67,0.6,0.35'})
    # a = manager.createInstance(2,{})
    # x = threading.Thread(target=a.run)
    # x.start()
    # x.join()
    insts = manager.runBatch(2,[{
        '--closures':{'rules': [ { 
      'name': 'sept_23_nov_11', 
      'conditionType': 'afterDays', 
      'threshold': 0, 
      'threshold2': 0, 
      'openAfter': 49, 
      'closeAfter': -1, 
      'parameter': 0,
      'locationTypesToClose': [6,22]
    }]}},
    {
        '--closures':{'rules': [ { 
      'name': 'sept_23_nov_1', 
      'conditionType': 'afterDays', 
      'threshold': 0, 
      'threshold2': 0, 
      'openAfter': 39, 
      'closeAfter': -1, 
      'parameter': 0,
      'locationTypesToClose': [6,22]
    }]}},
    {
        '--closures':{'rules': [ { 
      'name': 'sept_23_nov_21', 
      'conditionType': 'afterDays', 
      'threshold': 0, 
      'threshold2': 0, 
      'openAfter': 59, 
      'closeAfter': -1, 
      'parameter': 0,
      'locationTypesToClose': [6,22]
    }]}}
        ])
    print(f'finished {insts[0].completed}')

if __name__ == "__main__":
    main()