%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

%%

% Population in the simulator
Np = 179500;

RESULT = "Result_2024-02-08_11-10_T7_randref_aggr_Finalized";
R = readtimetable("/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/" + RESULT + "/A.xls");
R.Date.Format = 'uuuu-MM-dd';

Start_Date = datetime(2020,10,01);
End_Date = datetime(2021,01,31);
End_Date = Start_Date + 300;

R = R(isbetween(R.Date,Start_Date,End_Date),:);

N = height(R) - 1;

%%

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(string(fp.dir) + "/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");

K = Epid_Par.GetK;
P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,Start_Date,End_Date),:);

Q("Future",:) = [];

%% Construct optimization

import casadi.*

[f,~,~,J] = mf_epid_ode_model_SLPIA(Np);

% -----------------------------------
% Reference values

L_ref = R.L(2:end)';
I_ref = R.I(2:end)';
A_ref = R.A(2:end)';

% -----------------------------------
% Initial guess

x_PanSim = R(:,vn_SLPIA).Variables';
x0 = x_PanSim(:,1);
x_guess = x_PanSim(:,2:end) + randn(size(x_PanSim)-[0 1]);
x_fh = @(x_var) [x0 , x_var];

beta0 = R.TrRate(1);
beta_min = 0.01;
beta_max = 0.5;
beta_guess = R.TrRate(2:end-1)';
beta_guess = beta_guess + randn(size(beta_guess))*0.05;
beta_guess = min(max(beta_min,beta_guess),beta_max);
beta_fh = @(beta_var) [ beta0 , beta_var , beta_var(end) ];

% -----------------------------------
% Create optimization problem

helper = Pcz_CasADi_Helper('SX');

x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0,'ub',Np);
x = x_fh(x_var);

beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);
beta = beta_fh(beta_var);

tauL_guess = P.Param(1,K.L_iPeriod);
tauL = helper.new_var('tauL',[1,1],'lb',1/15,'ub',1);

tauP_guess = P.Param(1,K.P_iPeriod);
tauP = helper.new_var('tauP',[1,1],'lb',1/15,'ub',1);

tauA_guess = P.Param(1,K.A_iPeriod);
tauA = helper.new_var('tauA',[1,1],'lb',1/15,'ub',1);

pI_guess = P.Param(1,K.Pr_I);
pI = helper.new_var('pI',[1,1],'lb',0.01,'ub',0.99);

Params = SX(P.Param);
Params(:,K.L_iPeriod) = tauL;
% Params(:,K.P_iPeriod) = tauP;
% Params(:,K.A_iPeriod) = tauA;
Params(:,K.Pr_I) = pI;

% Enforce the state equations
for i = 1:N
    x_kp1 = f.Fn(x(:,i),Params(i,:)',beta(i),0);
    helper.add_eq_con( x_kp1 - x(:,i+1) );
end

% Minimize the tracking error
helper.add_obj('L_error',sumsqr(x_var(J.L,:) - L_ref),1);
helper.add_obj('I_error',sumsqr(x_var(J.I,:) - I_ref),1);
helper.add_obj('A_error',sumsqr(x_var(J.A,:) - A_ref),1);


helper.add_obj('beta_slope',sumsqr(diff(beta)),1e6);

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;
sol_guess = full(Fn_var(x_guess,beta_guess,tauL_guess,tauP_guess,tauA_guess,pI_guess));

% Solve the control optimization problem
ret = NL_solver.solve([],sol_guess);
% ret = NL_solver.solve();

% Get the solution
beta_sol = helper.get_value('beta');
x_sol = helper.get_value('x');

R.TrRate_Rec = beta_fh( beta_sol )';
x_Rec = x_fh( x_sol );

tauL = helper.get_value('tauL')
tauP = helper.get_value('tauP')
pI = helper.get_value('pI')

%%

Fig = figure;
delete(Fig.Children)

nexttile
plot(R.Date,[R.TrRate , R.TrRate_Rec])

SLPIA = vn_SLPIA;
for i = 1:J.nx
    nexttile
    plot(R.Date,[x_PanSim(i,:)' , x_Rec(i,:)'])
    title(SLPIA(i))
end
