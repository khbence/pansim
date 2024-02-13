%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. January 23. (2023a)

clear all

% Population in the simulator
Np = 179500;
% Np = 178746;

Conservativity_Mtp = 0.95;

%% Load parameters

Q = readtable("matlab/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");

P = Epid_Par.Get(Q);
Q("Future",:) = [];

Start_Date = datetime(2020,08,15);
End_Date = Q("Delta",:).Date;
P(P.Date < Start_Date,:) = [];

N = days(End_Date - Start_Date);
t_sim = 0:N;
d_sim = Start_Date + t_sim;

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

%% Reference signal

Mtp = 1;
Shift = 0;

Tp = 1;
Nr_Periods = 183;
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

N_MPC = Tp * Nr_Periods;

%%%
% Initial guess

Idx = 2:N_MPC+1;
Iref_k = Iref(Idx)';
Param_k = P.Param(Idx,:)';
Rel_beta_k = P.Rel_beta_Var(Idx,:)';

L_guess = Iref_k*0.34;
P_guess = Iref_k*0.56;
I_guess = Iref_k*0.98;
A_guess = Iref_k*1.23;
S_guess = Np - cumsum(L_guess);
x_guess = [
    S_guess
    L_guess
    P_guess
    I_guess
    A_guess
    ];
x_fh = @(x0,x_var) [x0 , x_var];

%%%
%  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
%                         0 0 0 0 0 0 0 , 1 1 1 ... ]
idx = reshape(ones(Tp,1) * (1:Nr_Periods),N_MPC,1);
I = eye(Nr_Periods);
M = I(:,idx);

beta_guess = ones(1,Nr_Periods) * 0.33;
beta_fh = @(beta_var) beta_var * M;

% -----------------------------------
% Create optimization problem

helper = Pcz_CasADi_Helper('SX');

p_Iref = helper.new_par('p_Iref',size(Iref_k),1);
p_Par = helper.new_par('p_Par',size(Param_k),1);
p_x0 = helper.new_par('p_x0',[J.nx,1],1);
p_MinMax_beta = helper.new_par('p_MinMax_beta',[1,2],1);
p_Rel_beta = helper.new_par('p_Rel_beta',size(Rel_beta_k),1);

x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0);
x = [p_x0 , x_var];

beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',p_MinMax_beta(1),'ub',p_MinMax_beta(2));
beta = beta_fh(beta_var) .* p_Rel_beta;

helper.add_obj('I_error',sumsqr(x_var(J.I,:) - p_Iref),1);

% Enforce the state equations
for k = 1:N_MPC
    x_kp1 = f.Fn(x(:,k),p_Par(:,k),beta(k),0);
    helper.add_eq_con( x_kp1 - x(:,k+1) );
end

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;

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

% Ax = [Ax nexttile]; hold on
% title('Beta')

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

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now);
mkdir(DIR)

for k = 0:N-N_MPC-1

    %%%
    % Current reference

    Idx = (1:N_MPC) + Tp*k;    
    NL_solver.set_param(p_Par,P.Param(Idx,:)');
    NL_solver.set_param(p_Iref,Iref(Idx+1)');
    NL_solver.set_param(p_x0,xx(:,Tp*k+1));
    NL_solver.set_param(p_MinMax_beta,[beta_min beta_max]);
    NL_solver.set_param(p_Rel_beta,P.Rel_beta_Var(Idx)')

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
    sol_guess = full(Fn_var(x_guess,beta_guess));

    %%%
    % Solve the control optimization problem
    ret = NL_solver.solve(helper.p_val,sol_guess);
    % ret = NL_solver.solve();


    %%%
    % Get the solution
    beta_sol = helper.get_value('beta');
    x_sol = helper.get_value('x');

    [~,Idx_Closest] = min(abs(T.Beta - beta_sol(1) * Conservativity_Mtp));
    PM(k+1,:) = T(Idx_Closest,Rstr_Vars).Variables;

    Pl_Iprd.YData(Idx) = x_sol(J.I,:);
    Pl_Bprd.YData(Idx) = beta_fh(beta_sol) .* P.Rel_beta_Var(Idx)';
    
    % -----------------------------------
    % Simulate and collect measurement

    r = Results(Tp*k+1,:);
    for d = 1:Tp
        simout = obj.runForDay(string(PM(k+1,:)));
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

    % Update LUT:
    Idx = Tp*k+1:Tp*k+Tp;
    beta_new = mean(beta_msd(Idx) ./ P.Rel_beta_Var(Idx));
    if isfinite(beta_new) && beta_new > 0
        Mtp = beta_new / T.Beta(Idx_Closest);
        T.Beta = T.Beta * (9 + Mtp)/10;
        T.Beta(Idx_Closest) = beta_new;
        beta_min = min(T.Beta);
        beta_max = max(T.Beta);
    end

end

filename = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now) + ".xls";
writetable(splitvars(Results),filename,"Sheet","Results");

exportgraphics(fig,strrep(filename,".xls",".pdf"),'ContentType','vector');

% clear all
