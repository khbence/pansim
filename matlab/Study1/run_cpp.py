import mexPanSim as sp

simulator = sp.SimulatorInterface()
init_options = ['panSim', '-r', ' ', '--diags','2', '--quarantinePolicy', '0', '-k', '0.00041',      '--progression', 'inputConfigFiles/progressions_Jun17_tune/transition_config.json',      '-A', 'inputConfigFiles/agentTypes_3.json', '-a', 'inputRealExample/agents1.json', '-l', 'inputRealExample/locations0.json',      '--infectiousnessMultiplier', '0.98,1.81,2.11,2.58,4.32,6.8,6.8', '--diseaseProgressionScaling', '0.94,1.03,0.813,0.72,0.57,0.463,0.45', '--closures','inputConfigFiles/emptyRules.json']
# '/home/reguly/pansim/tmpdirs_gergo3/400/argsFor--closures.json','--testingProbabilities', '0.00005,0.2,0.04,0.04,0.005,0.05']
simulator.initSimulation(init_options)


run_options = ['SONONE']

#run_options = ['TPdef', 'SO3','QU3','MA0.8','CF2000-0500']
results_agg = []
for i in range(0,140):
  results = simulator.runForDay(run_options)
  results_agg.append(results)
print(results_agg)
