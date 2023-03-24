function run_cpp()
    % Check if the mex exists
    dir = fileparts(mfilename('fullpath'));
    %if ~isequal(fileparts(which('example_mex')), dir)
        % Compile the mex
        cwd = cd(dir);
        cleanup_obj = onCleanup(@() cd(cwd));
        fprintf('Compiling example_mex\n');
        mex cpp_example_mex.cpp
    %end
    
    % Use the standard interface
    % This interface can be used for any mex interface function using the
    % pattern:
    %   Construction -    obj = mexfun('new',         ...)
    %   Destruction -           mexfun('delete', obj)
    %   Other methods - [...] = mexfun('method', obj, ...)
    % The standard interface avoids the need to write a specific interface
    % class for each mex file.
    fprintf('Using the standard interface\n');
    obj = mex_interface_cpp(str2fun([dir '/cpp_example_mex'])); % str2fun allows us to use the full path, so the mex need not be on our path
    
    inp_params = ["opt1", "opt2", "opt3"];
    whos inp_params
    class(inp_params)
    obj.initSimulation(inp_params);
    val_params = obj.runForDay(inp_params)

    clear obj % Clear calls the delete method
end
