%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

%%

clear all

% Population in the simulator
Np = 179500;
% Np = 178746;

%% Load parameters

[T,Rstr_Vars] = load_policy_measures;

Q = readtable("matlab/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");

P = Epid_Par.Get(Q);
Q("Future",:) = [];

% Start_Date = datetime(2020,08,15);
Start_Date = datetime(2020,09,23);
P(P.Date < Start_Date,:) = [];

% Erre mar nem nincs szukseg:
P.Rel_beta_Var = P.Pattern * Q.beta0 / Q.beta0(1);

%% Load policy measures

beta_min = min(T.Beta);
beta_max = max(T.Beta);

k0_Policy_Idx = 1;
k0_Policy_measures = T(k0_Policy_Idx,Rstr_Vars).Variables;
k0_Expected_beta = T.Beta(k0_Policy_Idx);

%%

Tp = 21;
[D,S] = Aggregator_B(Rstr_Vars,Tp);

Mtp = 0.3;
Shift = 0;

N = 16*3*7;
Nr_Periods = N / Tp;

t_sim = 0:N;
d_sim = Start_Date + t_sim;
d_sim.Format = "uuuu-MM-dd";
t_interp = t_sim + Shift;

REF_FILE = load("matlab/res2.mat");
x_ref = [1:length(REF_FILE.Ihat_daily); REF_FILE.Ihat_daily*Np*Mtp]';

spline_Iref = spline(x_ref(:, 1), x_ref(:, 2)); % create the funcion of the reference curve
spline_Irefd = fnder(spline_Iref, 1);           % calculate the first derivate
spline_Irefdd = fnder(spline_Iref, 2);          % calculate the second derivate

Ioff = ppval(spline_Iref, t_interp).';

%% Generate a random smooth reference signal

% 0, 1, 3, 6, 14
Rng_Int = 0;

if Rng_Int == 0
    Iref = Ioff;
else
    rng(Rng_Int)    
    while true
        
        Possible_Tks = divisors(N);
        Possible_Tks(Possible_Tks <= 14) = [];
        Possible_Tks(Possible_Tks > 70) = [];
        Tk = Possible_Tks( floor(( rand * (numel(Possible_Tks)-eps) ))+1 );
        
        Max_Inf = Np / 30;
        
        hyp = {};
        hyp.X = t_sim( sort(randperm(numel(t_sim),N/Tk)) )';
        hyp.y = rand(size(hyp.X)) * Max_Inf;
        hyp.sf = 36;
        hyp.sn = 25;
        hyp.ell = Tk;
        
        GP_eval(hyp)
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
end

figure(1), hold off
plot(d_sim,[Ioff Iref])

%% Design controller

import casadi.*

[f,~,~,J] = mf_epid_ode_model_SLPIA(Np);

%%

% Measured state
xx = nan(J.nx,N+1);

% Policy measures
PM = repmat(k0_Policy_measures,[N+1,1]);

% Measured beta
beta_msd = nan(1,N+1);

%%

% Load PanSim arguments
PanSim_args = load_PanSim_args;

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

simout = obj.runForDay(string(PM(1,:)));
[x0,~] = get_SLPIAb(simout,Np);
xx(:,1) = x0;

Iref = Iref + x0(J.I);

%%

SLPIA = ["S" "L" "P" "I" "A"];
simout2table_ = @(simout) simout2table(simout,'Start_Date',Start_Date,'Getter',@get_SLPIAb,'VariableNames',SLPIA);

R = [ simout2table_(simout) policy2table(k0_Policy_measures) table(k0_Expected_beta,k0_Expected_beta,'VariableNames',{'TrRateCmd','TrRateExp'}) ];
R = repmat(R,[N+1,1]);
R.Properties.RowTimes = d_sim;
R.TrRateBounds = R.TrRate*0 + [beta_min beta_max];
R.TrRate(2:end) = NaN;
R.I(2:end) = NaN;
R.Ioff = Ioff;
R.Iref = Iref;

z = zeros(height(R),1);
R = addvars(R,z,z,z,'NewVariableNames',{'k','d','TrRate_Mtp'});

Pred = R(:,[SLPIA,"TrRate"]);

Visualize_MPC(R,Pred,0);

%%

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now) + "_T" + num2str(Tp) + "_randref_aggr";
mkdir(DIR)

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
    S_guess = Np - cumsum(L_guess);
    x_guess = [
        S_guess
        L_guess
        P_guess
        I_guess
        A_guess
        ];
    x_fh = @(x_var) [xx(:,Tp*k+1) , x_var];
    
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
    
    lb = beta_guess*0 + beta_min;
    ub = beta_guess*0 + beta_max;
    d_MPC = Start_Date + (k:Nr_Periods-1)*Tp;
    S_ = retime(S,d_MPC);

    lb_ = S_.min_TrRate;
    ub_ = S_.max_TrRate;

    std_ = S_.std_TrRate;
    lb_ = S_.mean_TrRate - 2*std_;
    ub_ = S_.mean_TrRate + 2*std_;

    ldx = ismissing(lb_) | ismissing(ub_);
    lb_(ldx) = lb(ldx);
    ub_(ldx) = ub(ldx);

    beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',lb_,'ub',ub_);
    beta = beta_fh(beta_var);
    
    % Enforce the state equations
    for i = 1:N_MPC
        x_kp1 = f.Fn(x(:,i),P.Param(k+i,:)',beta(i),0);
        helper.add_eq_con( x_kp1 - x(:,i+1) );
    end
    
    % Minimize the tracking error
    helper.add_obj('I_error',(x_var(J.I,:) - Iref_k).^2, 1./(1:N_MPC));
    
    % Construct the nonlinear solver object
    NL_solver = helper.get_nl_solver("Verbose",true);
    
    % Retain the mapping for the free variables, which allows to construct an
    % initial guess vector for the nonlinear solver.
    Fn_var = NL_solver.helper.gen_var_mfun;
    sol_guess = full(Fn_var(x_guess,beta_guess));

    % Solve the control optimization problem
    ret = NL_solver.solve([],sol_guess);
    % ret = NL_solver.solve();

    % Get the solution
    beta_sol = helper.get_value('beta');
    x_sol = helper.get_value('x');

    % Update planned policy measures
    idx_sol = 1;
    for j = k:Nr_Periods-1

        Date0 = Start_Date + j*Tp;
        idx = find(S.Date == Date0,1);

        Sj = S(idx,:);
        if isempty(Sj)
            continue
        end

        Dj = D(isbetween(D.Date,Sj.Date1,Sj.Date2),:);
        [PMj,TrRatesj,DjU,mtp] = Intervention_Selector(T,Dj,Date0,Tp,beta_sol(idx_sol)-std_(idx_sol),Rstr_Vars);

        PM(j+1,:) = PMj;
        Rp = policy2table(PMj);
        for d = 1:Tp
            Idx = Tp*j+d;
            R(Idx,Rp.Properties.VariableNames) = Rp;
            R.TrRateCmd(Idx) = beta_sol(idx_sol);
            R.TrRateExp(Idx) = TrRatesj(d);
            R.TrRateBounds(Idx,:) = [lb_(idx_sol) ub_(idx_sol)];
        end

        idx_sol = idx_sol + 1;
    end
    R = quantify_policy(R);

    % Update prediction
    Pred(1:Tp*k+1,:) = array2table(Pred(1:Tp*k+1,:).Variables * NaN);
    Idx = Tp*k+1:height(Pred)-1;
    Pred.TrRate(Idx) = beta_fh(beta_sol)';
    Pred(Idx+1,SLPIA) = array2table(x_sol');
    
    % -----------------------------------
    % Simulate and collect measurement

    for d = 1:Tp
        simout = obj.runForDay(string(PM(k+1,:)));

        Idx = Tp*k+d;

        [O,simx,simbeta] = simout2table_(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
        R.k(Idx) = k;
        R.d(Idx) = d;

        fig = Visualize_MPC(R,Pred,Idx+1);
    
        xx(:,Tp*k+d+1) = simx;
    
        drawnow

        exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,Tp*k+d) + ".png")
    end
    x0 = simx;    
end

filename = DIR + "/A.xls";
writetimetable(R,filename,"Sheet","Results");
exportgraphics(fig,DIR + "/Fig.pdf",'ContentType','vector');

movefile(DIR,DIR+"_Finalized")

return
%%

movefile(DIR,DIR+"_Terminated")
