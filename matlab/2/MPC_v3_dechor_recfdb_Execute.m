
N = 6*7*4;
t_sim = 0:N;

fp = pcz_mfilename(mfilename("fullpath"));
fname = fullfile(fp.dir,"Output","Summary_2024-03-09.xls");
opts = detectImportOptions(fname);
opts = setvartype(opts,Vn.policy,"categorical");
T = readtable(fname,opts);

Iq_k = T.Properties.VariableNames(startsWith(T.Properties.VariableNames,'Iq_'));
T.Iq = T(:,Iq_k).Variables;
T(:,Iq_k) = [];

T.Pmx = (1:height(T))';

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
