function [f,h,hvar,J,Init] = mf_epid_ode_model_6comp(Np)
arguments
    Np = 1
end
%%
%  File: mf_epid_ode_model.m
%  Directory: 4_gyujtemegy/11_CCS/2021_COVID19_analizis/study13_SNMPC_LTV_delta
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. October 12. (2021b)
%

import casadi.*

[~,np] = Par.GetK;
[s,p,~,p_Cas] = Par.Symbolic;

Cnt(0);
% State variables:
S = sym('S'); J.S = Cnt;
L = sym('L'); J.L = Cnt;
% P = sym('P'); J.P = Cnt;
I = sym('I'); J.I = Cnt;
% A = sym('A'); J.A = Cnt;
H = sym('H'); J.H = Cnt;
D = sym('D'); J.D = Cnt;
R = sym('R'); J.R = Cnt;
nx = Cnt - 1;
% -----
x = [S;L;I;H;D;R];

x_Cas = SX(nx,1);
for xi = x.'
    name = char(xi);
    x_Cas(J.(name)) = SX.sym(name);
end

% Osszevont parameterek
hat_rho = 1/(1/s.zeta + s.gamma/s.rhoI + (1-s.gamma)/s.rhoA);
hat_delta = (1/s.zeta + s.gamma/s.rhoI + s.delta*(1-s.gamma)/s.rhoA) * hat_rho;
hat_eta = s.gamma * s.eta;

% Time-dependent parameter: Nr. of vaccinated per day
nu = sym('nu');
nu_Cas = SX.sym('nu');

% Unknown time-dependent parameters:
beta = sym('beta');   % transmission rate
w = sym('omega');
u = [beta;w];
u_Cas = [SX.sym('beta');SX.sym('w')];

J.beta = 1;
J.w = 2;

dS = -beta*hat_delta*I*S/Np - nu*S + w*R;
dL = beta*hat_delta*I*S/Np - s.alpha*L;
dI = s.alpha*L - hat_rho*I;
dH = hat_rho*hat_eta*I - s.lambda*H;
dD = s.mu*s.lambda*H;
dR = hat_rho*(1-hat_eta)*I + (1-s.mu)*s.lambda*H + nu*S - w*R;

f_sym = [dS dL dI dH dD dR].';
assert(double(simplify(sum(f_sym))) == 0);

Cnt(0);
J.Daily_All = Cnt;
J.Daily_New = Cnt;
J.Rc = Cnt;
J.Rt = Cnt;

h_sym = [
    L + I + H
    beta * hat_delta * I * S / Np
    beta * hat_delta / hat_rho
    beta * hat_delta / hat_rho * S / Np
    ];

matlabFunction(f_sym,'File','Fn_SLIHDR_ode','Vars',{x,u,p,nu});
matlabFunction(h_sym,'File','Fn_SLIHDR_out','Vars',{x,u,p});

Ts = 1;

f = {};
f.desc = 'f(x,u,p,v)';
f.val = x_Cas + Ts*Fn_SLIHDR_ode(x_Cas,u_Cas,p_Cas,nu_Cas);
f.Fn = Function('f',{x_Cas,u_Cas,p_Cas,nu_Cas},{f.val},{'x','u','p','v'},{f.desc});

f.      Input_1 = x_Cas;
f.desc__Input_1 = 'State vector (x)';
f.      Input_2 = u_Cas;
f.desc__Input_2 = 'Input vector (u = [beta,w])';
f.      Input_3 = p_Cas;
f.desc__Input_3 = 'Parameter vector (p)';
f.      Input_4 = nu_Cas;
f.desc__Input_4 = 'Vaccination inputs';

h = {};
h.desc = '[all_inf,daily_new_inf,R0,Rt]';
h.val = Fn_SLIHDR_out(x_Cas,u_Cas,p_Cas);
h.Fn = Function('h',{x_Cas,u_Cas,p_Cas},{h.val},{'x','u','p'},{h.desc});

h.      Input_1 = x_Cas;
h.desc__Input_1 = 'State vector (x)';
h.      Input_2 = u_Cas;
h.desc__Input_2 = 'Input vector (u = [beta,w])';
h.      Input_3 = p_Cas;
h.desc__Input_3 = 'Parameter vector (p)';

h.desc__Output_dim1 = 'All infected';
h.desc__Output_dim2 = 'Daily new infected';
h.desc__Output_dim3 = 'Basic reproduction number (R0)';
h.desc__Output_dim4 = 'Time-dependent reproduction number (Rt)';

hvar = {};
hvar.val = zeros(size(h.val));
hvar.Fn = [];
vars = [x_Cas;u_Cas;p_Cas];
Jh = jacobian(h.val,vars);

[Sigma,Sigma_half] = Pcz_CasADi_Helper.create_sym('Sigma',numel(vars),1,'str','sym');

hvar.val = diag(Jh * Sigma * Jh');
hvar.Fn = Function('hvar',{x_Cas,u_Cas,p_Cas,Sigma_half},{hvar.val},{'x','u','theta','Sigma'},{'[Var_all_inf,Var_daily_new_inf,Var_Rc]'});

J.nx = numel(x);
J.nu = numel(u);
J.ny = numel(h.val);
J.np = np;

J.H_ref = 1;
J.DN_ref = 2;

Init.x0 = zeros(J.nx,1);
Init.x0(J.L) = 10;
Init.x0(J.I) = 30;

Init.beta0 = 1/3;

end
