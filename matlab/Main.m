%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2023. July 03. (2023a)
%
%% Load matrix, find closest string [OK]

matrix = readMatrixFromFile('Data/data.txt');

floatColumn = cell2mat(matrix(:, end));
floatColumn_MinMax = [
    min(floatColumn,[],'omitnan') 
    max(floatColumn,[],'omitnan') 
    ];

lambda = 0.8;
referenceValue = [1-lambda lambda] * floatColumn_MinMax;

stringColumns = matrix(:, 1:end-1);

[~, index] = min(abs(floatColumn - referenceValue));

closestStrings = stringColumns(index, :)

%% Compute beta

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mex_interface_cpp(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
inp_params = ["panSim", "-r", " ", "--quarantinePolicy", "0", "-k", "0.00041", ...
     "--progression", "inputConfigFiles/progressions_Jun17_tune/transition_config.json", ...
     "-A", "inputConfigFiles/agentTypes_3.json", "-a", "inputRealExample/agents1.json", "-l", "inputRealExample/locations0.json", ...
     "--infectiousnessMultiplier", "0.98,1.81,2.11,2.58,4.32,6.8,6.8", "--diseaseProgressionScaling", "0.94,1.03,0.813,0.72,0.57,0.463,0.45"];
obj.initSimulation(inp_params);

%%%
% Execute simulation
inp_params = ["TPdef", "SONONE", "CF2000-0500"];
inp_params = string(closestStrings);
val_params = obj.runForDay(inp_params);

% MIVEL KELL MEGHIVNI COMPUTE_BETA FUGGVENYT?
compute_beta(val_params)    

clear obj % Clear calls the delete method