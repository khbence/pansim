
N = 6*7*4;
t_sim = 0:N;

fp = pcz_mfilename(mfilename("fullpath"));
dirname = fullfile(fp.dir,"Output");
dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output";
fname = fullfile(dirname,"Summary_2024-03-09.xls");
CtrlDirName = "Ctrl_Sum2024-03-09";
opts = detectImportOptions(fname);
opts = setvartype(opts,Vn.policy,"categorical");
T = readtable(fname,opts);

Iq_k = T.Properties.VariableNames(startsWith(T.Properties.VariableNames,'Iq_'));
T.Iq = T(:,Iq_k).Variables;
T(:,Iq_k) = [];

T.Pmx = (1:height(T))';

%% Sigmoid reference

h = [1 2 1 2 1]*N/7;
u = @(i,a) zeros(1,h(i))+a;
S = @(i,a,b) Epid_Par.Sigmoid(a,b,h(i));
Iref = [ ...
    u(1,0) ...
    S(2,0,1) ...
    u(3,1) ...
    S(4,1,0) ...
    u(5,0), 0 ...
    ]'*900 + 100;

for Tp = [28,21,14,7,1]
    Name = "Sigmoid_T" + sprintf('%02d',Tp);
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,CtrlDirName,Name); end
end

return

%% Control goal: flatten the curve

FreeT = readtimetable('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-03-09/FreeSpread/FreeSpread_2024-03-14_08-56.xls');

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
fig = figure(123); 
delete(fig.Children)
ax = axes(fig);
hold on; grid on; box on;
plot(Date,[Iref,Ifree]);
plot(FreeT.Date,FreeT.I);
xlim(Date([1,end]))
ax.YLim(1) = 0;


for Tp = [28,21,14,7]
    Name = "Scenario_T" + sprintf('%02d',Tp);
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,CtrlDirName,Name); end
end

return

%%

for Tp = [21,1,14,7]
%%    
    Tp_str = sprintf('%02d',Tp);
    
    %________________________________________________________________
    %% Sigmoid reference
    
    h = [1 2 1 2 1]*N/7;
    u = @(i,a) zeros(1,h(i))+a;
    S = @(i,a,b) Epid_Par.Sigmoid(a,b,h(i));
    Iref = [ ...
        u(1,0) ...
        S(2,0,1) ...
        u(3,1) ...
        S(4,1,0) ...
        u(5,0), 0 ...
        ]'*900 + 100;
    
    Name = "Sigmoid_T" + Tp_str;
    save("Iref.mat","Iref","Tp","N","Name")
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,Name); end
    
    %________________________________________________________________
    %% Constant reference
    
    Iref = t_sim'*0 + 500;
    
    Name = "C500_T" + Tp_str;
    save("Iref.mat","Iref","Tp","N","Name")
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,Name); end
    
    %________________________________________________________________
    %% Increasing reference
    
    Iref = t_sim(:) ./ N * 1500;
    
    Name = "Lin1500_T" + Tp_str;
    save("Iref.mat","Iref","Tp","N","Name")
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,Name); end
    
    %________________________________________________________________
    %% Constant reference
    
    Iref = t_sim'*0 + 1000;
    
    Name = "C1000_T" + Tp_str;
    save("Iref.mat","Iref","Tp","N","Name")
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,Name); end

    %________________________________________________________________
    %% Random reference
    
    % 0, 1, 3, 6, 14, 1996
    % Rng_Int = round(rand*10000);
    % Rng_Int = 1996;
    % Rng_Int = 0;
    % Rng_Int = 3466;
    
    Name = "Ketpupu_Teve_T" + Tp_str;
    Rng_Int = 1647; % <---- 5 + 20 db szep eredmeny 2024.02.14. (február 14, szerda), 11:38
    
    Iref = hp.generate_path(Rng_Int,N);
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,Name); end
    
    %________________________________________________________________
    %% Random reference
    
    Name = "Erdekes_Teve_T" + Tp_str;
    Rng_Int = 7597; % <---- 3db szep eredmeny 2024.02.14. (február 14, szerda), 11:38
    
    Iref = hp.generate_path(Rng_Int,N);
    for i=1:20; MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,Name); end
    
end
