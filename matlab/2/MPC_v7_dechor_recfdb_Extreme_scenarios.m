%% Load LUT

% CnfName = "CtrlOmicron";
% RecBetaRange = [0.01,2.5];
% InfectiousnessMultiplier  = [0.98,2.58,2.58,2.58,4.32,6.8,6.8];
% DiseaseProgressionScaling = [0.94,0.72,0.57,0.72,0.57,0.463,0.45];
% Closures = "Scenario2.json";

CnfName = "CtrlAlpha55";
RecBetaRange = [0.01,1.8];
InfectiousnessMultiplier  = [0.98,1.81,2.58,2.58,4.32,6.8,6.8];
DiseaseProgressionScaling = [0.94,1.03,0.57,0.72,0.57,0.463,0.45];
Closures = "Scenario2.json";

PanSim_args = ps.load_PanSim_args("Manual", ...
    "InfectiousnessMultiplier",InfectiousnessMultiplier, ...
    "DiseaseProgressionScaling",DiseaseProgressionScaling, ...
    "Closures",Closures);

N = 6*7*4;
t_sim = 0:N;

fp = pcz_mfilename(mfilename("fullpath"));
dirname = fullfile(fp.dir,"Output");
dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output";
fname = fullfile(dirname,"Summary_2024-03-09.xls");
CtrlDirName = "Ctrl_Sum2024-05-30";
opts = detectImportOptions(fname);
opts = setvartype(opts,Vn.policy,"categorical");
T = readtable(fname,opts);

[~,Pmx_lezar] = max(T.IQ);
[~,Pmx_free] = min(T.IQ);
T.TrRate(Pmx_free) = max(T.TrRate) + 1e-4;

% Manual correction
T.TrRate(T.IQ == 411042) = 0.123;

[~,Idx] = sort(T.TrRate);
T = T(Idx,:);

Iq_k = T.Properties.VariableNames(startsWith(T.Properties.VariableNames,'Iq_'));
T.Iq = T(:,Iq_k).Variables;
T(:,Iq_k) = [];

T.Pmx = (1:height(T))';

%% Control goal: flatten the curve

FreeT = readtimetable('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-03-09/FreeSpread/FreeSpread_2024-03-14_08-56.xls');

N = 210;
t_sim = 0:N;

FreeMean = 72;
FreeStd = 24;
FreePeak = 3300;
Date = t_sim+C.Start_Date;
Ifree = normpdf(t_sim,FreeMean,FreeStd)';
Ifree = Ifree / max(Ifree) * FreePeak;

CtrlMean = FreeMean + 7*7;
CtrlStd = 48;
CtrlPeak = 1500;
Iref = normpdf(t_sim,CtrlMean,CtrlStd)';
Iref = Iref / max(Iref) * CtrlPeak;

CtrlMean = FreeMean + 8*7;
CtrlStd = 36;
CtrlPeak = 2500;
Iref3 = normpdf(t_sim,CtrlMean,CtrlStd)';
Iref3 = Iref3 / max(Iref3) * CtrlPeak;

CtrlMean = FreeMean - 3*7;
CtrlStd = 20;
CtrlPeak = 750;
Iref41 = normpdf(t_sim,CtrlMean,CtrlStd)';
Iref41 = Iref41 / max(Iref41) * CtrlPeak;

CtrlMean = FreeMean + 14*7;
CtrlStd = 30;
CtrlPeak = 1000;
Iref42 = normpdf(t_sim,CtrlMean,CtrlStd)';
Iref42 = Iref42 / max(Iref42) * CtrlPeak;

Iref4 = Iref41 + Iref42;
Iref5 = flip(Iref4);

% 2024.03.19. (március 19, kedd), 12:48
CtrlMean = FreeMean + 12*7;
CtrlStd = 48;
CtrlPeak = 1500;
Iref2 = normpdf(t_sim,CtrlMean,CtrlStd)';
Iref2 = Iref2 / max(Iref2) * CtrlPeak;
Iref2 = Iref2.^4;
Iref2 = Iref2 / max(Iref2) * 2000;

fig = figure(123); 
delete(fig.Children)
ax = axes(fig);
hold on; grid on; box on;
plot(Date,Iref,'DisplayName','Scenario 1','LineWidth',2);
plot(Date,Ifree,'DisplayName','Free spread');
plot(Date,Iref2,'DisplayName','Scenario 2','LineWidth',2);
plot(Date,Iref3,'DisplayName','Scenario 3');
plot(Date,Iref4,'DisplayName','Scenario 4','LineWidth',3);
plot(FreeT.Date,FreeT.I,'DisplayName','Free spread');
xlim(Date([1,end]))
ax.YLim(1) = 0;
legend

%%

for i = 1:50
%%
    for Tp = [7,14,21,30]
    %%
        Name = CnfName + "_Flatten_T" + sprintf('%02d',Tp);
        MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,Iref,CtrlDirName,Name, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-3);
    end
end

return

%%
%%
%%
%%

Tp = 7;
Name = CnfName + "_Scenario41_Free_T" + sprintf('%02d',Tp);
R = MPC_v6_dechor_recfdb_OneSimulation(T,Tp,9*7,Iref4,CtrlDirName,Name, ...
    "FreeSpreadFromDate",datetime(2020,12,20), ...
    "PanSimArgs",PanSim_args,"RecHorizonTp",-3);

%%

for i = 1:20
    Sim_interventions(T,R.Pmx(1:Tp:end),Tp,CtrlDirName,Name, ...
        "Visualize",true, ...
        "PanSimArgs",PanSim_args);
end

%%

Tp = 21;
Name = CnfName + "_Scenario3_Free_T" + sprintf('%02d',Tp);
MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,Iref3,CtrlDirName,Name, ...
    "FreeSpreadFromDate",C.Start_Date + 21*5, ...
    "PanSimArgs",PanSim_args,"RecHorizonTp",-3); 

%%

Tp = 21;
for i = 1:20
    Name = CnfName + "_Scenario2_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,Iref2,CtrlDirName,Name, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-3);
end

%%

Tp = 21;
for i = 1:20
    Name = CnfName + "_Scenario3_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,Iref3,CtrlDirName,Name, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-3);
end

%%

Tp = 14;
for i = 1:20
    Name = CnfName + "_Scenario4_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,Iref4,CtrlDirName,Name, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-6,"PunishOvershoot",false, ...
        "BetaMultiplier",true);
end

%%

Tp = 21;
for i = 1:20
    Name = CnfName + "_Scenario4_T" + sprintf('%02d',Tp) + "_NoUpdate";
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,Iref4,CtrlDirName,Name, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-6,"PunishOvershoot",false, ...
        "BetaMultiplier",false);
end

%%

for i = 1:30
%%
    Tp = 14;
    CnfName = "ContWithOmicron";
    PanSim_args = ps.load_PanSim_args("Omicron");

    Name = CnfName + "_Scenario4_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,[],CtrlDirName,Name, ...
        "PunishOvershoot",true,"Limit",1500,"BetaCost",1,"BetaSlopeCost",1,"InfCost",1, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-6,"BetaMultiplier",true);
end

%%

CnfName = "ContWithOmicron_noMtp";
PanSim_args = ps.load_PanSim_args("Omicron");

Tp = 14;
for i = 1:30
    Name = CnfName + "_Scenario4_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,[],CtrlDirName,Name, ...
        "PunishOvershoot",true,"Limit",1500,"BetaCost",1,"BetaSlopeCost",1,"InfCost",1, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-6,"BetaMultiplier",false);
end

% %%

CnfName = "NewScenario1";
PanSim_args = ps.load_PanSim_args("Scenario1");

Tp = 14;
for i = 1:30
    Name = CnfName + "_Scenario4_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,[],CtrlDirName,Name, ...
        "PunishOvershoot",true,"Limit",1500,"BetaCost",1,"BetaSlopeCost",1,"InfCost",1, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-3);
end

% %%

CnfName = "Simple";
PanSim_args = ps.load_PanSim_args("Free");

Tp = 14;
for i = 1:30
    Name = CnfName + "_Scenario4_T" + sprintf('%02d',Tp);
    MPC_v6_dechor_recfdb_OneSimulation(T,Tp,N,[],CtrlDirName,Name, ...
        "PunishOvershoot",true,"Limit",1500,"BetaCost",1,"BetaSlopeCost",1,"InfCost",1, ...
        "PanSimArgs",PanSim_args,"RecHorizonTp",-3);
end


