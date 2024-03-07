function run_cpp()
    % Check if the mex exists
    dir = fileparts(mfilename('fullpath'));
    obj = mexPanSim_wrap(str2func([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
    
    inp_params = ["panSim", "-r", " ", "--quarantinePolicy", "0", "-k", "0.00041", ...
         "--progression", "inputConfigFiles/progressions_Jun17_tune/transition_config.json", ...
         "-A", "inputConfigFiles/agentTypes_3.json", "-a", "inputRealExample/agents1.json", "-l", "inputRealExample/locations0.json", ...
         "--infectiousnessMultiplier", "0.98,1.81,2.11,2.58,4.32,6.8,6.8", "--diseaseProgressionScaling", "0.94,1.03,0.813,0.72,0.57,0.463,0.45"];
    whos inp_params
    class(inp_params)
    obj.initSimulation(inp_params);
    while true
        inp_params = ["TPdef", "SONONE", "CF2000-0500"];
        val_params = obj.runForDay(inp_params)
        %print(val_params)
    end

    clear obj % Clear calls the delete method
end
