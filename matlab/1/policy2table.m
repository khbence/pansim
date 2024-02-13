function T = policy2table(PM)
arguments
    PM
end

P = array2table(PM,'VariableNames',policy_varnames);
T = quantify_policy(P);