%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)
% 
% Decreasing, shrinking, narrowing, shortening horizon

%%

clear all

%% Load parameters

s = load("Iref.mat");

Iref = s.Iref;
Tp = s.Tp;
N = s.N;
Nr_Periods = N / Tp;

T = hp.load_policy_measures_2;

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2023-12-19_JN1.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q(["Transient","Original","Future"],:);
P = Epid_Par.Get(Q);

%%

[~,Pmx_Mindent_lezar] = max(vecnorm(T.Iq,1,2));
[~,Pmx_Mindent_felenged] = min(vecnorm(T.Iq,1,2));

%% First policy

k0_Pmx = max(T.Pmx);
k0_PM = T(k0_Pmx,Vn.policy).Variables;
k0_TrRateExp = T.Beta(k0_Pmx);

%%

Start_Date = C.Start_Date;
End_Date = Start_Date + N;
P = P(isbetween(P.Date,Start_Date,End_Date),:);

t_sim = 0:N;
d_sim = Start_Date + t_sim;

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

%%

R = [ ...
    hp.simout2table(simout) ...
    table(k0_Pmx,'VariableNames',"Pmx") ...
    hp.policy2table(k0_PM) ...
    table(k0_TrRateExp,k0_TrRateExp,NaN,NaN,'VariableNames',...
        {'TrRateCmd','TrRateExp','TrRateRec','Ipred'}) ...
    array2table(nan(size(Vn.SLPIAHDR)),"VariableNames",Vn.SLPIAHDR+"r")
    ];
R = repmat(R,[N+1,1]);
R = [R hp.param2table(P.Param)];
R.Properties.RowTimes = d_sim;
R.TrRate(2:end) = NaN;
R.I(2:end) = NaN;

Iref = Iref + R.I(1);
R.Iref = Iref;

z = zeros(height(R),1);
R = addvars(R,z,z,'NewVariableNames',{'k','d'});

beta_min = min(T.Beta);
beta_max = max(T.Beta);
R.TrRateBounds = repmat([beta_min beta_max],[height(R),1]);

beta_min_std = min(max(0,T.mean_TrRate - 2*T.std_TrRate));
beta_max_std = max(T.mean_TrRate + 2*T.std_TrRate);
R.TrRateRange = repmat([beta_min_std beta_max_std],[height(R),1]);

Visualize_MPC(R,0,"Tp",max(Tp,7));

%%

h = [2 2 3]*N/7;
wIerr = [ ones(1,h(1)) , Epid_Par.Sigmoid(1,0,h(2)) , zeros(1,h(3)) ] + 0.1;

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
% DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/Result_" + string(Now) + "_T" + num2str(Tp) + "_randref_aggr";
% mkdir(DIR)

Actuator_Overshoot = 0;

for k = 0:Nr_Periods-1

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
    x0 = R(Tp*k+1,Vn.SLPIAHDR).Variables';
    x_fh = @(x_var) [x0 , x_var];
    
    %%%
    %  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
    %                         0 0 0 0 0 0 0 , 1 1 1 ... ]
    idx = reshape(ones(Tp,1) * (1:Nr_Periods-k),N_MPC,1);
    I = eye(Nr_Periods-k);
    M = I(:,idx);
    
    beta_guess = ones(1,Nr_Periods-k) * 0.33;
    % beta_overshoot = Actuator_Overshoot * 2.^(-1:-1:k-Nr_Periods);
    % beta_fh = @(beta_var) ( beta_var + beta_overshoot ) * M;
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

    % Correct beta solution based on the previous actuation error
    % beta_sol(1) = min(max(beta_min,beta_sol(1) - 0.2*Actuator_Overshoot),beta_max);

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
    R = hp.quantify_policy(R);

    % Update prediction
    % Pred(1:Tp*k+1,:) = array2table(Pred(1:Tp*k+1,:).Variables * NaN);
    Idx = Tp*k+1:height(R);
    R.Ipred(Idx) = x(J.I,:)';
    
    % -----------------------------------
    % Simulate and collect measurement

    for d = 1:Tp
        simout = obj.runForDay(string(R(Tp*k+d,Vn.policy).Variables));

        Idx = Tp*k+d;

        O = hp.simout2table(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
        R.k(Idx) = k;
        R.d(Idx) = d;

        fig = Visualize_MPC(R,Idx+1,"Tp",max(Tp,7));
    
        drawnow

        % exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,Tp*k+d) + ".png")
    end

    Pend = (k+1)*Tp;
    if Pend >= 7
        R = rec_SLPIAHDR(R,Start_Date + [0,Pend],C.Np,'WeightBetaSlope',1e4);
        % Actuator_Overshoot = R.TrRateRec(Pend) - beta_sol(1);
    end
end

fig = Visualize_MPC(R,N+1,"Tp",max(Tp,7));

% writetimetable(R,DIR + "/A.xls","Sheet","Result");
% exportgraphics(fig,DIR + "/A_.pdf",'ContentType','vector');

exportgraphics(fig,C.DIR_GenLUT + "/" + s.Name + "_" + string(Now) + ".pdf",'ContentType','vector');
exportgraphics(fig,C.DIR_GenLUT + "/" + s.Name + "_" + string(Now) + ".jpg",'ContentType','vector');

R = rec_SLPIAHDR(R,'WeightBetaSlope',1e4);
writetimetable(R,C.DIR_GenLUT + "/" + s.Name + "_" + string(Now) + ".xls","Sheet","Result");

% movefile(DIR,DIR+"_Finalized")
