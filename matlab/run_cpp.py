import mexPanSim as sp

simulator = sp.SimulatorInterface()
init_options = ['panSim', '-r', ' ', '--quarantinePolicy', '0', '-k', '0.00041',      '--progression', 'inputConfigFiles/progressions_Jun17_tune/transition_config.json',      '-A', 'inputConfigFiles/agentTypes_3.json', '-a', 'inputRealExample/agents1.json', '-l', 'inputRealExample/locations0.json',      '--infectiousnessMultiplier', '0.98,1.81,2.11,2.58,4.32,6.8,6.8', '--diseaseProgressionScaling', '0.94,1.03,0.813,0.72,0.57,0.463,0.45']
simulator.initSimulation(init_options)


run_options = ['TPdef', 'SONONE', 'CF2000-0500']
results = simulator.runForDay(run_options)