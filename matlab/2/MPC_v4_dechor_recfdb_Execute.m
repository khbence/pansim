
N = 6*7*4;
t_sim = 0:N;

fp = pcz_mfilename(mfilename("fullpath"));
dirname = fullfile(fp.dir,"Output");
dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output";
fname = fullfile(dirname,"Summary_2024-03-09.xls");
CtrlDirName = "Ctrl_Sum2024-04-19";
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

Tp = 21;
Name = "Scenario41_Free_T" + sprintf('%02d',Tp);
MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref4,CtrlDirName,Name, ...
    "FreeSpreadFromDate",datetime(2020,12,20)); 

%%

Tp = 21;
Name = "Scenario3_Free_T" + sprintf('%02d',Tp);
MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref3,CtrlDirName,Name, ...
    "FreeSpreadFromDate",C.Start_Date + 21*5); 

%%

Tp = 30;
for i = 1:20
    Name = "Scenario1_T" + sprintf('%02d',Tp);
    MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,CtrlDirName,Name);
end

%%

Tp = 21;
for i = 1:20
    Name = "Scenario4_T" + sprintf('%02d',Tp);
    MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref4,CtrlDirName,Name);
end

% %%

Tp = 21;
for i = 1:20
    Name = "Scenario2_T" + sprintf('%02d',Tp);
    MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref2,CtrlDirName,Name);
end

% %%

Tp = 21;
for i = 1:20
    Name = "Scenario3_T" + sprintf('%02d',Tp);
    MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref3,CtrlDirName,Name);
end

%%

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