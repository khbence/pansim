function PM = policy2table(PM)
arguments
    PM
end

if ~istable(PM)
    PM = array2table(PM,'VariableNames',Vn.policy);
end
PM = hp.quantify_policy(PM);