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

Mtp = 1.5;
Shift = 0;

P = 21;
Nr_Periods = 16; 8;
N = P*Nr_Periods; % days
t_sim = 0:N;
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
    cellstr(k0_Policy_measures)'
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
Pl_Imsd = plot(t_sim,xx(J.I,:),'DisplayName','PanSim');
Pl_Iref = plot(t_sim,Iref,'DisplayName','reference');
Pl_Iprd = plot(t_sim,Iref + nan,'DisplayName','predicted (SLPIA)');
Leg = legend('Location','northeast');

Ax = [Ax nexttile]; hold on
title('Beta')
Pl_Bmsd = plot(t_sim,beta_msd,'DisplayName','measured');
Pl_Bcmd = plot(t_sim,beta_cmd,'DisplayName','commanded');
Pl_Bprd = plot(t_sim,beta_msd,'DisplayName','predicted');
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

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_" + string(Now);
mkdir(DIR)

for k = 0:Nr_Periods-1

    N_MPC = P * (Nr_Periods-k);

    %%%
    % Current reference

    Idx = P*k+1:N;
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
    x_fh = @(x_var) [xx(:,P*k+1) , x_var];
    
    z_guess = Iref_k*0;

    %%%
    %  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
    %                         0 0 0 0 0 0 0 , 1 1 1 ... ]
    idx = reshape(ones(P,1) * (1:Nr_Periods-k),N_MPC,1);
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
    
    z_var = helper.new_var('z',size(z_guess),1,'str','full');
    z = [0 z_var];

    helper.add_obj('I_error',sumsqr(x_var(J.I,:) - Iref_k),1);
    helper.add_obj('z_error',sumsqr(z),1);
    
    % Enforce the state equations
    for i = 1:N_MPC
        x_ip1 = f.Fn(x(:,i),p,beta(i),0);
        z_ip1 = z(1,i) + Iref_k(i) - x_var(J.I,i);
        helper.add_eq_con( x_ip1 - x(:,i+1) );
        helper.add_eq_con( z_ip1 - z(1,i+1) );
    end
    
    % Construct the nonlinear solver object
    NL_solver = helper.get_nl_solver("Verbose",true);
    
    % Retain the mapping for the free variables, which allows to construct an
    % initial guess vector for the nonlinear solver.
    Fn_var = NL_solver.helper.gen_var_mfun;
    sol_guess = full(Fn_var(x_guess,beta_guess,z_guess));

    % Solve the control optimization problem
    ret = NL_solver.solve([],sol_guess);
    % ret = NL_solver.solve();

    % Get the solution
    beta_sol = helper.get_value('beta');
    x_sol = helper.get_value('x');

    [~,Idx_Closest] = min(abs(T.Beta - beta_sol(1)));
    PM(k+1,:) = T(Idx_Closest,Rstr_Vars).Variables;

    Pl_Iprd.YData = [ nan(1,P*k) x0(J.I) x_sol(J.I,:) ];
    Pl_Bprd.YData = [ nan(1,P*k) beta_fh(beta_sol) nan ];
    
    % -----------------------------------
    % Simulate and collect measurement

    r = Results(P*k+1,:);
    for d = 1:P
        simout = obj.runForDay(string(PM(k+1,:)));
        [simx,simbeta] = get_SLPIAb(simout,Np);

        xx(:,P*k+d+1) = simx;
        beta_msd(P*k+d) = simbeta;
        beta_cmd(P*k+d) = beta_sol(1);
    
        r.Day = P*k+d;
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

        exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,P*k+d) + ".png")
    end
    x0 = simx; 
    beta0 = simbeta;

    % Update LUT:
    beta_new = mean(beta_msd(P*k+1:P*k+P));
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
