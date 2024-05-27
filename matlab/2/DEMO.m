
PM = [ "TP035" "PL0" "CF2000-0500" "SO3" "QU2" "MA0.8" ];

%%%
% Create simulator object
fp = pcz_mfilename(mfilename('fullpath'));
DIR = fp.dir;

pansim = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
pansim.initSimulation(ps.load_PanSim_args);

pansim2 = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
pansim2.initSimulation(ps.load_PanSim_args);

for i = 1:1
    simout = pansim.runForDay(PM);
    simout = pansim2.runForDay(PM);
end

