function R_ = rec_SLPIAHDR(R,DSpan,Np)
arguments
    R_,DSpan

    % Population in the simulator
    Np = 179500;
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 12. (2023a)

%%

Start_Date = DSpan(1);
End_Date = DSpan(2);

ldx = isbetween(R.Date,Start_Date,End_Date);
R_ = R;

R = R(ldx,:);

N = height(R) - 1;

%% Construct optimization

import casadi.*

K = Epid_Par.GetK;
[f,~,~,J] = mf_epid_ode_model_8comp(Np);

% -----------------------------------
% Reference values

D = R.D1 + R.D2;
H = R.I5h + R.I6h;
S = Np - D - R.IMM1 - R.L - R.P - R.I - R.A - H;

x_PanSim = [S R.L R.P R.I R.A H D R.IMM1]';

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

w_guess = 0.001;
w = helper.new_var('w',[1,1],'lb',1e-6,'ub',0.01);

IdxLbUb = [
    K.tauL   , 1/7  , 1
    K.tauP   , 1/7  , 1
    K.tauI   , 1/13 , 1
    K.tauA   , 1/13 , 1
    K.tauH   , 1/13 , 1
    K.pI     , 0.01 , 0.99
    K.pH     , 0.01 , 0.99
    K.pD     , 0.01 , 0.99
    ];

p_guess = P.Param(1,IdxLbUb(:,1));
p = helper.new_var('p',[1 height(IdxLbUb)],'lb',IdxLbUb(:,2),'ub',IdxLbUb(:,2));

Params = SX(P.Param);
for i = 1:height(IdxLbUb)
    Params(:,IdxLbUb(i,1)) = p(i);
end

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
helper.add_obj('R_error',sumsqr(x(J.R,:) - R.IMM1'),0.01);

helper.add_obj('beta_slope',sumsqr(diff(beta)),1e5);

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;
sol_guess = full(Fn_var(x_guess,beta_guess,p_guess,w_guess));

% Solve the control optimization problem
ret = NL_solver.solve([],sol_guess);

% Get the solution
beta_sol = helper.get_value('beta');
x_sol = helper.get_value('x');
p_sol = helper.get_value('p');
w = helper.get_value('w');

R.TrRate_Rec = beta_fh( beta_sol )';
x = x_fh( x_sol );

R(:,vn_SLPIAHDR) = array2table(x');

R_(ldx,:) = R(:,R.Properties.VariableNames);

