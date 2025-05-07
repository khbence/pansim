%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

clear all

% Population in the simulator
Np = 179500;
% Np = 178746;

%% Load parameters

Q = readtable("matlab/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");

P = Epid_Par.Get(Q);
Q("Future",:) = [];

% Start_Date = datetime(2020,08,15);
Start_Date = datetime(2020,09,23);
P(P.Date < Start_Date,:) = [];

P.Rel_beta_Var = P.Pattern * Q.beta0 / Q.beta0(1);

%% Load policy measures

[T,Rstr_Vars] = load_policy_measures;

Beta_MinMax = [
    min(T.Beta)
    max(T.Beta)
    ];

beta_min = min(T.Beta);
beta_max = max(T.Beta);

k0_Policy_Idx = 1;
k0_Policy_measures = T(k0_Policy_Idx,Rstr_Vars).Variables;
k0_Expected_beta = T.Beta(k0_Policy_Idx);

%% Reference signal

Mtp = 1;
Shift = 0;

Tp = 7;
Nr_Periods = 24;
N = Tp*Nr_Periods; % days
t_sim = 0:N;
d_sim = Start_Date + t_sim;
d_sim.Format = "uuuu-MM-dd";
t_interp = t_sim + Shift;

REF_FILE = load("matlab/res2.mat");
x_ref = [1:length(REF_FILE.Ihat_daily); REF_FILE.Ihat_daily*Np*Mtp]';

spline_Iref = spline(x_ref(:, 1), x_ref(:, 2)); % create the funcion of the reference curve
spline_Irefd = fnder(spline_Iref, 1);           % calculate the first derivate
spline_Irefdd = fnder(spline_Iref, 2);          % calculate the second derivate

Iref = ppval(spline_Iref, t_interp).';
dIref = ppval(spline_Irefd, t_interp).';
ddIref = ppval(spline_Irefdd, t_interp).';

%% Design controller

import casadi.*

[f,~,~,J] = mf_epid_ode_model_SLPIA(Np);

%%

% Measured state
xx = nan(J.nx,N+1);

simout = zeros(1,49);
simx = zeros(1,J.nx);

Table_cell = repmat([
    {0}
    k0_Policy_measures'
    {simout}
    {simx}
    {0}
    {0}
    ]',[N,1]);

VarNames = ["Day",Rstr_Vars,"simout","simx","cmdbeta","simbeta"];
Results = cell2table(Table_cell,'VariableNames',VarNames);


% Policy measures
PM = repmat(k0_Policy_measures,[N+1,1]);

% Derivative of measured I
Diff_N = 10;
dI = [ dIref(1:Diff_N+1)' nan(1,N-Diff_N)];

% Measured beta
beta_msd = nan(1,N+1);

% Commanded beta
beta_cmd = [ T.Beta(k0_Policy_Idx) nan(1,N) ];

%% Initialize visualization

fig = figure(13);
Tl = tiledlayout(2,1,"TileSpacing","tight","Padding","tight");

Ax = nexttile; hold on
title('Infected')
Pl_Imsd = plot(d_sim,xx(J.I,:),'DisplayName','PanSim');
Pl_Iref = plot(d_sim,Iref,'DisplayName','reference');
Pl_Iprd = plot(d_sim,Iref + nan,'DisplayName','predicted (SLPIA)');
Leg = legend('Location','northeast');

Ax = [Ax nexttile]; hold on
title('Beta')
Pl_Bmsd = plot(d_sim,beta_msd,'DisplayName','measured');
Pl_Bcmd = plot(d_sim,beta_cmd,'DisplayName','commanded');
Pl_Bprd = plot(d_sim,beta_msd,'DisplayName','predicted');
Leg = legend('Location','northeast');

Link_XLim = linkprop(Ax,'XLim');

for ax = Ax
    grid(ax,'on')
    box(ax,'on')
end

% return

%%

% Load PanSim arguments
PanSim_args = load_PanSim_args;

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

simout = obj.runForDay(string(PM(1,:)));
[x0,beta0] = get_SLPIAb(simout,Np);
xx(:,1) = x0;

%%

create_row = @(simout,PM,TrRateExp) [
    simout2table(simout,'Start_Date',Start_Date,'Getter',@get_SLPIAb,'VariableNames',["S" "L" "P" "I" "A"]) ...
    policy2table(PM) ...
    table(TrRateExp,'VariableNames',{'TrRateExp'})
    ];

R = repmat(create_row(simout,k0_Policy_measures,k0_Expected_beta),[N+1,1]);
R.Properties.RowTimes = d_sim;

Pred = R(:,["S","L","P","I","A","TrRate"]);

%%

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now);
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
    
    beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min*P.Rel_beta_Var(k+1),'ub',beta_max*P.Rel_beta_Var(k+1));
    beta = beta_fh(beta_var);
    
    helper.add_obj('I_error',sumsqr(x_var(J.I,:) - Iref_k),1);
    
    % Enforce the state equations
    for i = 1:N_MPC
        x_kp1 = f.Fn(x(:,i),P.Param(k+i,:)',beta(i),0);
        helper.add_eq_con( x_kp1 - x(:,i+1) );
    end
    
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

    [~,Idx_Closest] = min(abs(T.Beta - beta_sol(1)));
    PM(k+1,:) = T(Idx_Closest,Rstr_Vars).Variables;

    Pl_Iprd.YData = [ nan(1,Tp*k) x0(J.I) x_sol(J.I,:) ];
    Pl_Bprd.YData = [ nan(1,Tp*k) beta_fh(beta_sol) nan ];

    Pred.TrRate(1:end-1) = beta_fh(beta_sol)';
    Pred.S(2:end) = x_sol(J.S,:)';
    Pred.L(2:end) = x_sol(J.L,:)';
    Pred.P(2:end) = x_sol(J.P,:)';
    Pred.I(2:end) = x_sol(J.I,:)';
    Pred.A(2:end) = x_sol(J.A,:)';
    
    % -----------------------------------
    % Simulate and collect measurement

    r = Results(Tp*k+1,:);
    for d = 1:Tp
        simout = obj.runForDay(string(PM(k+1,:)));

        R(Tp*k+d+1,:) = create_row(simout,PM(k+1,:),T.Beta(Idx_Closest));

        % Visualize_MPC(T);

        [simx,simbeta] = get_SLPIAb(simout,Np);

        xx(:,Tp*k+d+1) = simx;
        beta_msd(Tp*k+d) = simbeta;
        beta_cmd(Tp*k+d) = beta_sol(1);
    
        r.Day = Tp*k+d;
        r(:,Rstr_Vars) = cell2table(PM(k+1,:),'VariableNames',Rstr_Vars);
        r.simout = simout;
        r.simx = simx(:)';
        r.cmdbeta = beta_sol(1);
        r.simbeta = simbeta;

        Results(r.Day,:) = r;

        % Update plot
        Pl_Imsd.YData = xx(J.I,:);
        Pl_Bmsd.YData = beta_msd;
        Pl_Bcmd.YData = beta_cmd;
    
        drawnow

        exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,Tp*k+d) + ".png")
    end
    x0 = simx; 
    beta0 = simbeta;

    Pred(1:Tp,:) = [];

    % Update LUT:
    % beta_new = mean(beta_msd(Tp*k+1:Tp*k+Tp));
    % if isfinite(beta_new) && beta_new > 0
    %     Mtp = beta_new / T.Beta(Idx_Closest);
    %     T.Beta = T.Beta * (29 + Mtp)/30;
    %     T.Beta(Idx_Closest) = beta_new;
    %     beta_min = min(T.Beta);
    %     beta_max = max(T.Beta);
    % end

end

filename = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now) + ".xls";
writetable(splitvars(Results),filename,"Sheet","Results");

exportgraphics(fig,strrep(filename,".xls",".pdf"),'ContentType','vector');

% clear all
