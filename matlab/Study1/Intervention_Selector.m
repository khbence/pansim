function [PM,TrRates,DsU,mtp] = Intervention_Selector(T,Dk,Date0,Tp,TrRate_ref,Rstr_Vars,args)
arguments
    T, Dk;
    Date0;
    Tp;
    TrRate_ref;
    Rstr_Vars;
    args.Window_Radius = 7;
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 07. (2023a)
%

t = 0:Tp-1;
% d = Date0 + t;

Ds = [];

for i = 0:Tp-1
    Di = Dk(isbetween(Dk.Date, ...
        Date0+i - args.Window_Radius, ...
        Date0+i + args.Window_Radius),:);
    Dsi = groupsummary(Di,Rstr_Vars,["min","max","mean","std"],"TrRate");
    Dsi.Idx = zeros(height(Dsi),1) + i;

    if isempty(Ds)
        Ds = Dsi;
    else
        Ds = [Ds;Dsi];
    end
end

DsU = Ds(:,[Rstr_Vars,"min_TrRate","Idx"]);
DsU = unstack(DsU,"min_TrRate","Idx",'VariableNamingRule','modify');

T_DsU = outerjoin(T,DsU);

mtp = ones(1,Tp);

NewVariableNames = setdiff(DsU.Properties.VariableNames,Rstr_Vars);
i = 1;
for var = NewVariableNames
    mtp(i) = mean(T_DsU.(var) ./ T_DsU.Beta,'omitmissing');
    T_DsU.(var) = T_DsU.Beta * mtp(i);
    i = i + 1;
end

TrRate = T_DsU(:,NewVariableNames).Variables;

[~,idx] = min( vecnorm(TrRate - TrRate_ref,2,2) );

PM = T_DsU(idx,Rstr_Vars + "_T").Variables;

TrRates = TrRate(idx,:);

end
