%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

% Population in the simulator
Np = 179500;
% Np = 178746;

%% Load parameters

Q = readtable("Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q("Original",:);

% Get ordial numbers of the parameters used in every model
[K,np] = Epid_Par.GetK;

p = zeros(np,1);
p(K.tauL) = 1 / Q.Period_L;
p(K.tauP) = 1 / Q.Period_P;
p(K.tauI) = 1 / Q.Period_I;
p(K.tauA) = 1 / Q.Period_A;
p(K.tauH) = 1 / Q.Period_H;
p(K.Rel_beta_A) = Q.Rel_beta_A;
p(K.pI) = Q.Pr_I;
p(K.pH) = Q.Pr_H;
p(K.pD) = Q.Pr_D;

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

%% Reference signal

Mtp = 4;
Shift = 0;

Nr_Weeks = 24;
N = 7*Nr_Weeks; % days
t_sim = 0:N;
t_interp = t_sim + Shift;

REF_FILE = load("matlab/res2.mat");
x_ref = [1:length(REF_FILE.Ihat_daily); REF_FILE.Ihat_daily*Np*Mtp]';

spline_Iref = spline(x_ref(:, 1), x_ref(:, 2)); % create the funcion of the reference curve
spline_Irefd = fnder(spline_Iref, 1);           % calculate the first derivate
spline_Irefdd = fnder(spline_Iref, 2);          % calculate the second derivate

Iref = ppval(spline_Iref, t_interp);
dIref = ppval(spline_Irefd, t_interp);
ddIref = ppval(spline_Irefdd, t_interp);

%% Initial guess

L_guess = Iref*0.15;
P_guess = Iref*0.201;
I_guess = Iref*0.401;
A_guess = Iref*0.401;
S_guess = Np - cumsum(L_guess);
x_guess = [
    S_guess
    L_guess
    P_guess
    I_guess
    A_guess
    ];
x0 = x_guess(:,1)*1.15;
x_guess(:,1) = [];
x_fh = @(x_var) [x0 , x_var];

%%%
%  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
%                         0 0 0 0 0 0 0 , 1 1 1 ... ]
idx = reshape(ones(7,1) * (1:Nr_Weeks),N,1);
I = eye(Nr_Weeks);
M = I(:,idx);

beta_guess = ones(1,Nr_Weeks) * 0.33;
beta_fh = @(beta_var) beta_var * M;

%% Design controller

import casadi.*

[f,~,~,J] = mf_epid_ode_model_SLPIA(Np);

helper = Pcz_CasADi_Helper('SX');

x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0);
x = x_fh(x_var);

beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);
beta = beta_fh(beta_var);

helper.add_obj('I_error',sumsqr(x(J.P,:) + x(J.I,:) + x(J.A,:) - Iref),1);

% Enforce the state equations
for k = 1:N
    x_kp1 = f.Fn(x(:,k),p,beta(k),0);
    helper.add_eq_con( x_kp1 - x(:,k+1) );
end

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;

sol_guess = full(Fn_var(x_guess,beta_guess));

% ret = NL_solver.solve([],sol_guess);
ret = NL_solver.solve();

x_val = x_fh(helper.get_value('x'));
beta_val = beta_fh(helper.get_value('beta'));

figure(1), plot([sum(x_val([J.P J.I J.A],:),1) ; Iref]')
figure(2), plot(beta_val)

return

%%

% Measured state
nx = 4;
x = nan(nx,N+1);

simout_prev = zeros(1,49);
simx_prev = zeros(1,nx);

Table_cell = repmat([
    {0}
    k0_Policy_measures'
    {simout_prev}
    {simout_prev}
    {simx_prev}
    {simx_prev}
    {0}
    {0}
    ]',[N,1]);

VarNames = ["Day",Rstr_Vars,"simout0","simout1","simx0","simx1","cmdbeta","simbeta"];
Results = cell2table(Table_cell,'VariableNames',VarNames);


% Policy measures
PM = repmat(k0_Policy_measures,[N+1,1]);

% Derivative of measured I
Diff_N = 10;
dI = [ dIref(1:Diff_N+1)' nan(1,N-Diff_N)];

% Measured beta
beta_msd = nan(1,N+1);

% Computed beta, which is possibly not implementable
beta_cmp = [ T.Beta(k0_Policy_Idx) nan(1,N) ];

% Commanded beta
beta_cmd = [ T.Beta(k0_Policy_Idx) nan(1,N) ];

% Integrator's state
Int_e = [ 0 nan(1,N) ];

% Parameters for numerical differentiation of measured I
Polyfit_ord = 3;
Polyfit_t = -Diff_N+1:0;

% Two parameters of the SEIR model
k2 = 0.37;
k3 = 0.1429;

%% Initialize visualization

fig = figure(13);
Tl = tiledlayout(4,1,"TileSpacing","tight","Padding","tight");

Ax = nexttile; hold on
title('Infected')
Pl_Imsd = plot(t_sim,x(3,:));
Pl_Iref = plot(t_sim,Iref);

Ax = [Ax nexttile]; hold on
title('Derivative of infected')
Pl_dImsd = plot(t_sim,dI);
Pl_dIref = plot(t_sim,dIref);

Ax = [Ax nexttile]; hold on
title('Integrator''s state')
Pl_Int = plot(t_sim,Int_e);

Ax = [Ax nexttile]; hold on
title('Beta')
Pl_Bmsd = plot(t_sim,beta_msd,'DisplayName','measured');
Pl_Bcmd = plot(t_sim,beta_cmd,'DisplayName','commanded');
Pl_Bcmp = plot(t_sim,beta_cmp,'DisplayName','controller prescribed');
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
% dir = fileparts(mfilename('fullpath'));
% obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
% obj.initSimulation(PanSim_args);

for k = 2:N+1

    % -----------------------------------
    % Simulate and collect measurement

    % simout = obj.runForDay(string(PM(k-1,:)));
    % [simx,simbeta] = get_SEIRb(simout,Np);

    x(:,k) = simx;
    beta_msd(k-1) = simbeta;

    S = simx(1);
    E = simx(2);
    I = simx(3);
    R = simx(4);

    % Accumulate error
    e = Iref(k) - I;
    Int_e(k) = Int_e(k-1) + e;

    % Compute the derivative of the MEASURED I
    if k >= Diff_N+1
        Polyfit_I = x(3,Polyfit_t+k);
        p = polyfit(Polyfit_t,Polyfit_I,Polyfit_ord);
        dI(k) = p(end-1); % polyval(polyder(p),0)
    end

    % Compute the derivative of the REFERENCE I
    % if k >= Diff_N+1
    %     Polyfit_I = Iref(Polyfit_t+k);
    %     p = polyfit(Polyfit_t,Polyfit_I,Polyfit_ord);
    %     dIref(k) = p(end-1); % polyval(polyder(p),0)
    % end

    % -----------------------------------
    % Log simulation data

    r = Results(k-1,:);

    r.Day = k-1;
    r(:,Rstr_Vars) = cell2table(PM(k-1,:),'VariableNames',Rstr_Vars);
    r.simout0 = simout_prev;
    r.simout1 = simout;
    r.simx0 = simx_prev;
    r.simx1 = simx(:)';
    r.cmdbeta = beta_cmd(k-1);
    r.simbeta = simbeta;

    Results(k-1,:) = r;

    simout_prev = simout;
    simx_prev = simx;

    % -----------------------------------
    % Compute control input

    de = dIref(k) - dI(k);

    v = kI * Int_e(k) + kP * e + kD * de;
    u = (ddIref(k) + E*(k2^2 + k3*k2) - I*k3^2 + v) / (I*S*k2) * Np;

    beta_cmp(k) = u;
    beta_cmd(k) = min(max(beta_min,u),beta_max);

    % Unit-windup compensation
    Int_e(k) = Int_e(k) + ( beta_cmd(k) - beta_cmp(k) ) * k_anti_windup;

    if e * Int_e(k-1) < 0
        % Reset integrator
        Int_e(k) = Int_e(k) * k_reset_integrator;
    end

    [~,Idx_Closest] = min(abs(T.Beta - beta_cmd(k)));
    PM(k,:) = T(Idx_Closest,Rstr_Vars).Variables;

    % Update plot
    Pl_Imsd  .YData = x(3,:);
    Pl_Iref  .YData = Iref;
    Pl_dImsd .YData = dI;
    Pl_dIref .YData = dIref;
    Pl_Int   .YData = Int_e;
    Pl_Bmsd  .YData = beta_msd;
    Pl_Bcmp  .YData = beta_cmp;
    Pl_Bcmd  .YData = beta_cmd;

    drawnow
end

Results.dI = dI(2:end)';
Results.Iref = Iref(2:end);
Results.dIref = Iref(2:end);
Results.ddIref = Iref(2:end);

Now = datetime;
Now.Format = "uuuu-MM-dd_HH:mm";
filename = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now) + ".xls";
writetable(splitvars(Results),filename,"Sheet","Results");
writetable(table(kI,kP,kD,k_anti_windup,k_reset_integrator),filename,"Sheet","Parameters")

% clear all
