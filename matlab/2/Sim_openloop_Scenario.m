%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 29. (2023a)
% 

% CnfName = "Omicron70_7days";
% RecBetaRange = [0.01,2.5];
% InfectiousnessMultiplier = [0.98,2.58,2.58,2.58,4.32,6.8,6.8];
% DiseaseProgressionScaling = [0.94,0.72,0.57,0.72,0.57,0.463,0.45];
% Closures = "Scenario2.json";

CnfName = "Alpha70_7days";
RecBetaRange = [0.01,1.8];
InfectiousnessMultiplier  = [0.98,1.81,2.58,2.58,4.32,6.8,6.8];
DiseaseProgressionScaling = [0.94,1.03,0.57,0.72,0.57,0.463,0.45];
Closures = "Scenario2.json";

% CnfName = "StartOmicron";
% RecBetaRange = [0.01,3];
% InfectiousnessMultiplier  = [2.58,1,1,1,1,1,1];
% DiseaseProgressionScaling = [0.72,1,1,1,1,1,1];
% Closures = "emptybbRules.json";

% Load PanSim arguments
PanSim_args = ps.load_PanSim_args("Manual", ...
    "InfectiousnessMultiplier",InfectiousnessMultiplier, ...
    "DiseaseProgressionScaling", DiseaseProgressionScaling, ...
    "Closures",Closures);

% Simulate interventions of:
xlsname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-04-19/Scenario1_T30/2024-04-19_17-21_typical.xls";

% Store the results here:
% /home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/
DirName = "Ctrl_Sum2024-05-30";
Name = CnfName + "__Sc1typical_T30";

% Simulate
for i = 1:10 
    Sim_interventions_by_xls(xlsname,PanSim_args,DirName,Name);
end
