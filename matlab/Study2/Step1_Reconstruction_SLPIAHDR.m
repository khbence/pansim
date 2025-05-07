%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

%%

% Population in the simulator
Np = C.Np;

%{
RESULT = "Result_2024-02-13_16-59_T28_allcomb";
R = readtimetable("/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/" + RESULT + "/A47.xls");
R.Date.Format = 'uuuu-MM-dd';

Start_Date = datetime(2020,10,01);
% End_Date = Start_Date + 300;
End_Date = datetime(2021,01,31);

R = R(isbetween(R.Date,Start_Date,End_Date),:);
%}

%%

Start_Date = R.Date(1);
End_Date = R.Date(end);

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

[f,~,~,J] = epid.ode_model_8comp(Np);

% -----------------------------------
% Reference values

D = double(R.D1 + R.D2);
H = double(R.I5h + R.I6h);
S = Np - D - double(R.IMM1) - R.L - R.P - R.I - R.A - H;

x_PanSim = double([S R.L R.P R.I R.A H D R.IMM1]');

% -----------------------------------
% Initial guess

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
tauL = helper.new_var('tauL',[1,1],'lb',1/3,'ub',1);

tauP_guess = P.Param(1,K.P_iPeriod);
tauP = helper.new_var('tauP',[1,1],'lb',1/3,'ub',1);

tauI_guess = P.Param(1,K.I_iPeriod);
tauI = helper.new_var('tauI',[1,1],'lb',1/5,'ub',1);

tauA_guess = P.Param(1,K.A_iPeriod);
tauA = helper.new_var('tauA',[1,1],'lb',1/5,'ub',1);

tauH_guess = P.Param(1,K.H_iPeriod);
tauH = helper.new_var('tauH',[1,1],'lb',1/14,'ub',1/7);

pI_guess = P.Param(1,K.Pr_I);
pI = helper.new_var('pI',[1,1],'lb',0.01,'ub',0.99);

pH_guess = P.Param(1,K.Pr_H);
pH = helper.new_var('pH',[1,1],'lb',0.01,'ub',0.99);

pD_guess = P.Param(1,K.Pr_D);
pD = helper.new_var('pD',[1,1],'lb',0.01,'ub',0.99);

w_guess = 0.001;
w = helper.new_var('w',[1,1],'lb',1e-6,'ub',0.01);

Params = SX(P.Param);
Params(:,K.L_iPeriod) = tauL;
Params(:,K.P_iPeriod) = tauP;
Params(:,K.I_iPeriod) = tauI;
Params(:,K.A_iPeriod) = tauA;
Params(:,K.H_iPeriod) = tauH;
Params(:,K.Pr_I) = pI;
Params(:,K.Pr_I) = pH;
Params(:,K.Pr_I) = pD;

% Enforce the state equations
for i = 1:N
    x_kp1 = f.Fn(x(:,i),Params(i,:)',beta(i),0,w);
    helper.add_eq_con( x_kp1 - x(:,i+1) );
end

% Minimize the tracking error
helper.add_obj('S_error',sumsqr(x(J.S,:) - S'),0.01);
helper.add_obj('L_error',sumsqr(x(J.L,:) - R.L'),1);
helper.add_obj('P_error',sumsqr(x(J.P,:) - R.P'),1);
helper.add_obj('I_error',sumsqr(x(J.I,:) - R.I'),1);
helper.add_obj('A_error',sumsqr(x(J.A,:) - R.A'),1);
helper.add_obj('H_error',sumsqr(x(J.H,:) - H'),100);
helper.add_obj('D_error',sumsqr(x(J.D,:) - D'),10);
helper.add_obj('R_error',sumsqr(x(J.R,:) - double(R.IMM1)'),0.01);

% helper.add_obj('State_error',sumsqr(x - x_PanSim));


helper.add_obj('beta_slope',sumsqr(diff(beta)),1e5);

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;
sol_guess = full(Fn_var(x_guess,beta_guess, ...
    tauL_guess,tauP_guess,tauI_guess,tauA_guess,tauH_guess, ...
    pI_guess,pH_guess,pD_guess, ...
    w_guess));

% Solve the control optimization problem
ret = NL_solver.solve([],sol_guess);
% ret = NL_solver.solve();

% Get the solution
beta_sol = helper.get_value('beta');
x_sol = helper.get_value('x');

R.TrRate_Rec = beta_fh( beta_sol )';
x_Rec = x_fh( x_sol );

tauL = 1/helper.get_value('tauL')
tauP = 1/helper.get_value('tauP')
tauI = 1/helper.get_value('tauI')
tauA = 1/helper.get_value('tauA')
tauH = 1/helper.get_value('tauH')
pI = helper.get_value('pI')
pH = helper.get_value('pH')
pD = helper.get_value('pD')
w = helper.get_value('w')

%%

Fig = figure;
delete(Fig.Children)

nexttile
plot(R.Date,[R.TrRate , R.TrRate_Rec])
title('Transmission rate')

for i = 1:J.nx
    nexttile
    plot(R.Date,[x_PanSim(i,:)' , x_Rec(i,:)'])
    title(Vn.SLPIAHDR(i))
end
