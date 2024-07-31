
function R = pcz_table_mergevars(R)
    
    % Get the variable names from the read table
    varNames = R.Properties.VariableNames;
    
    % Find all variables that end with "_1"
    splitVars = varNames(endsWith(varNames, '_1'));
    
    for i = 1:numel(splitVars)
        % Extract the base name of the variable (remove the "_1" suffix)
        baseName = splitVars{i}(1:end-2);
        
        % Find all parts of this split variable
        partVars = varNames(startsWith(varNames, baseName + "_"));
        
        R.(baseName) = R(:,partVars).Variables;
        R = movevars(R,baseName,"Before",partVars{1});
        R(:,partVars) = [];
    end

end

