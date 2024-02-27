function PM = policy2table(PM)
arguments
    PM
end

if ~istable(PM)
    PM = array2table(PM,'VariableNames',Vn.policy);
end
PM = Vn.quantify_policy(PM);