%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 26. (2023a)
% 
% Decreasing, shrinking, narrowing, shortening horizon. 
% Feedback using the reconstructed, estimated epidemic state.

%%

clear all

%% Load parameters

% anal.Generate_LUT_1

T = hp.load_policy_measures_2;

fp = pcz_mfilename(mfilename('fullpath'));

Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2024-02-26_Agens_Wild.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
P = Epid_Par.Get(Q);

%%

[~,Pmx_Mindent_lezar] = max(vecnorm(T.Iq,1,2));
[~,Pmx_Mindent_felenged] = min(vecnorm(T.Iq,1,2));

%% First policy

k0_Pmx = max(T.Pmx);
k0_PM = T(k0_Pmx,Vn.policy).Variables;
k0_TrRateExp = T.Beta(k0_Pmx);

%%

Tp = 14;

N = 6*7*4;
Nr_Periods = N / Tp;

Start_Date = C.Start_Date;
End_Date = Start_Date + N;
P = P(isbetween(P.Date,Start_Date,End_Date),:);

t_sim = 0:N;
d_sim = Start_Date + t_sim;

%% Generate a random smooth reference signal

% 0, 1, 3, 6, 14, 1996
Rng_Int = round(rand*10000);
% Rng_Int = 1996;
% Rng_Int = 0;
% Rng_Int = 3466;
Rng_Int = 1647; % <---- 5 + 20 db szep eredmeny 2024.02.14. (február 14, szerda), 11:38
% Rng_Int = 7597; % <---- 3db szep eredmeny 2024.02.14. (február 14, szerda), 11:38

rng(Rng_Int)    
while true
    
    Possible_Tks = divisors(N);
    Possible_Tks(Possible_Tks <= 10) = [];
    Possible_Tks(Possible_Tks > 70) = [];
    Tk = Possible_Tks( floor(( rand * (numel(Possible_Tks)-eps) ))+1 );
    
    Max_Inf = C.Np / 30;
    
    hyp = {};
    hyp.X = t_sim( sort(randperm(numel(t_sim),N/Tk)) )';
    hyp.y = rand(size(hyp.X)) * Max_Inf;
    hyp.sf = 36;
    hyp.sn = 25;
    hyp.ell = Tk;
    
    hpy = GP_eval(hyp);
    Iref = GP_eval(hyp,t_sim);
    
    wFnSup = 2.5;
    x = linspace(-wFnSup,wFnSup,numel(t_sim));
    w = normpdf(x,0,1)';
    w = w - w(1);
    [mw,idx] = max(w);
    w = w ./ mw;
    w(idx:end) = 1;
    
    Iref = Iref .* w;
    if all(Iref >= 0)
        break
    end
end

% Iref = Iref*0 + 1500;
% Iref = t_sim(:) ./ N * 1500;
% 
% h = [1 2 1 2 1]*N/7;
% u = @(i,a) zeros(1,h(i))+a;
% S = @(i,a,b) Epid_Par.Sigmoid(a,b,h(i));
% Iref = [ ...
%     u(1,0) ...
%     S(2,0,1) ...
%     u(3,1) ...
%     S(4,1,0) ...
%     u(5,0), 0 ...
%     ]'*800;

figure(5), plot(d_sim,Iref)

%% Design controller

import casadi.*

[f,~,~,J] = epid.ode_model_8comp(C.Np);

%%

% Measured state
% xx = nan(J.nx,N+1);

% Policy measures
PM = repmat(k0_PM,[N+1,1]);

% Measured beta
beta_msd = nan(1,N+1);

%%

% Load PanSim arguments
PanSim_args = ps.load_PanSim_args;

%%%
% Create simulator object
DIR = fileparts(mfilename('fullpath'));
obj = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

simout = obj.runForDay(string(k0_PM));

%% Initialize timetable `R` (Results)

% Create a first row for `R`
R = [ ...
    hp.simout2table(simout)             ... PanSim output
    table(k0_Pmx,'VariableNames',"Pmx") ... Ordinal no. of applied policy measure (Pmx)
    hp.policy2table(k0_PM)              ... Applied policy measures in flags and values
    table(k0_TrRateExp,k0_TrRateExp,NaN,NaN,'VariableNames', ... Further variables:
        {'TrRateCmd','TrRateExp','TrRateRec','Ipred'})       ... Ipred: legacy
    array2table(nan(1,Nr_Periods),'VariableNames',Vn.IQk(0:Nr_Periods-1)) ... Planned IQ in the different phases
    array2table(nan(1,Nr_Periods),'VariableNames',Vn.TrRatek(0:Nr_Periods-1)) ... Estimated TrRate in the different phases
    array2table(nan(1,Nr_Periods),'VariableNames',Vn.Ipredk(0:Nr_Periods-1)) ... Predicted I in the different phases
    array2table(nan(1,Nr_Periods),'VariableNames',Vn.Hpredk(0:Nr_Periods-1)) ... Predicted H in the different phases
    array2table(nan(size(Vn.SLPIAHDR)),"VariableNames",Vn.SLPIAHDR+"r") ... Reconstructed state
    ];

% We assume that the initial state of the epidemic is known, therefore, the first
% reconstructed state is the actual state
R(:,Vn.SLPIAHDR + "r") = R(:,Vn.SLPIAHDR);

% Construct the full table by repeating the first row 
R = repmat(R,[N+1,1]);

% Append parameters to `R` as new colums
R = [R hp.param2table(P.Param)];

% Update the time flags of the timetable
R.Properties.RowTimes = d_sim;

% Remove values, which are not known yet
R.TrRate(2:end) = NaN;
R.I(2:end) = NaN;
R.Ir(2:end) = NaN;
% .... there would be more, but those are not relevant

% Append the reference trajectory to `R` as a new column
Iref = Iref + R.I(1);
R.Iref = Iref;

% Append `k` (control term) and `d` (day) to `R` as new columns
z = zeros(height(R),1);
R = addvars(R,z,z,'NewVariableNames',{'k','d'});

% Append the bounds for the transmission rate
beta_min = min(T.Beta);
beta_max = max(T.Beta);
R.TrRateBounds = repmat([beta_min beta_max],[height(R),1]);

% Append the estimated range of the transmission rate
beta_min_std = min(max(0,T.mean_TrRate - 2*T.std_TrRate));
beta_max_std = max(T.mean_TrRate + 2*T.std_TrRate);
R.TrRateRange = repmat([beta_min_std beta_max_std],[height(R),1]);

Visualize_MPC_v3(R,0,0,"Tp",max(Tp,7));

%%

h = [2 2 3]*N/7;
wIerr = [ ones(1,h(1)) , Epid_Par.Sigmoid(1,0,h(2)) , zeros(1,h(3)) ] + 0.1;

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
% DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/Result_" + string(Now) + "_T" + num2str(Tp) + "_randref_aggr";
% mkdir(DIR)

for k = 0:Nr_Periods-1
    % This is the `k`th control term, which simulates `Tp` days

    N_MPC = Tp * (Nr_Periods-k);

    %%%
    % Current reference

    Idx = Tp*k+1:N;
    Iref_k = Iref(Idx+1)';

    %%%
    % Initial guess
    
    L_guess = Iref_k*0.15;
    P_guess = Iref_k*0.201;
    I_guess = Iref_k*0.401;
    A_guess = Iref_k*0.401;
    S_guess = C.Np - cumsum(L_guess);
    x_guess = [
        S_guess
        L_guess
        P_guess
        I_guess
        A_guess
        A_guess*0 % H
        A_guess*0 % D
        A_guess*0 % R
        ];
    x0 = R(Tp*k+1,Vn.SLPIAHDR + "r").Variables';
    x_fh = @(x_var) [x0 , x_var];
    
    %%%
    %  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
    %                         0 0 0 0 0 0 0 , 1 1 1 ... ]
    idx = reshape(ones(Tp,1) * (1:Nr_Periods-k),N_MPC,1);
    I = eye(Nr_Periods-k);
    M = I(:,idx);
    
    beta_guess = ones(1,Nr_Periods-k) * 0.33;
    beta_fh = @(beta_var) beta_var * M;

    % -----------------------------------
    % Create optimization problem

    helper = Pcz_CasADi_Helper('SX');

    x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0);
    x = x_fh(x_var);
    
    beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);
    beta = beta_fh(beta_var);
    
    % Enforce the state equations
    for i = 1:N_MPC
        x_kp1 = f.Fn(x(:,i),P.Param(k+i,:)',beta(i),0,0);
        helper.add_eq_con( x_kp1 - x(:,i+1) );
    end
    
    % Minimize the tracking error
    helper.add_obj('I_error',(x_var(J.I,:) - Iref_k).^2,wIerr(1:N_MPC));
    
    % Construct the nonlinear solver object
    NL_solver = helper.get_nl_solver("Verbose",true);
    
    % Retain the mapping for the free variables, which allows to construct an
    % initial guess vector for the nonlinear solver.
    Fn_var = NL_solver.helper.gen_var_mfun;
    sol_guess = full(Fn_var(x_guess,beta_guess));

    % Solve the control optimization problem
    ret = NL_solver.solve([],sol_guess);

    % Get the solution
    beta_sol = helper.get_value('beta');
    x_sol = helper.get_value('x');
    x = x_fh(x_sol);

    % Update planned policy measures
    idx_sol = 1;
    for j = k:Nr_Periods-1

        [~,Pmx] = min(abs(T.Beta - beta_sol(idx_sol)));

        Rp = T(Pmx,Vn.policy);
        for d = 1:Tp
            Idx = Tp*j+d;
            R.Pmx(Idx) = Pmx;
            R(Idx,Rp.Properties.VariableNames) = Rp;
            R.TrRateCmd(Idx) = beta_sol(idx_sol);
            R.TrRateExp(Idx) = T.Beta(Pmx);
        end

        idx_sol = idx_sol + 1;
    end
    R = Vn.quantify_policy(R);

    % Update prediction
    Idx = Tp*k+1:height(R);
    R.(Vn.Ipredk(k))(Idx) = x(J.I,:)';
    R.(Vn.Hpredk(k))(Idx) = x(J.H,:)';
    R.(Vn.TrRatek(k))(Idx) = R.TrRateCmd(Idx);
    R.(Vn.IQk(k))(Idx) = Vn.IQ(R(Idx,Vn.policy + "_Val"));
    
    % -----------------------------------
    % Simulate and collect measurement

    for d = 1:Tp
        simout = obj.runForDay(string(R(Tp*k+d,Vn.policy).Variables));

        Idx = Tp*k+d;

        O = hp.simout2table(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
        R.k(Idx) = k;
        R.d(Idx) = d;

        fig = Visualize_MPC_v3(R,Idx+1,k,"Tp",max(Tp,7));
    
        drawnow

        % exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,Tp*k+d) + ".png")
    end

    Pend = (k+1)*Tp;
    if Pend >= 7
        R = rec_SLPIAHDR(R,Start_Date + [0,Pend],C.Np,'WeightBetaSlope',1e4);
    else
        R(:,Vn.SLPIAHDR + "r") = R(:,Vn.SLPIAHDR);
    end
end

fig = Visualize_MPC_v3(R,N+1,Nr_Periods,"Tp",max(Tp,7));

% writetimetable(R,DIR + "/A.xls","Sheet","Result");
% exportgraphics(fig,DIR + "/Fig_.pdf",'ContentType','vector');

exportgraphics(fig,C.DIR_GenLUT + "/Fig_" + string(Now) + ".pdf",'ContentType','vector');
exportgraphics(fig,C.DIR_GenLUT + "/Fig_" + string(Now) + ".jpg",'ContentType','vector');

R = rec_SLPIAHDR(R,'WeightBetaSlope',1e4);
writetimetable(R,C.DIR_GenLUT+"/A_"+string(Now)+".xls","Sheet","Result");

% movefile(DIR,DIR+"_Finalized")
