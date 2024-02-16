function [f,h,hvar,J] = ode_model_8comp(Np)
arguments
    Np = 9709786
end
%%
%  File: mf_epid_ode_model.m
%  Directory: 4_gyujtemegy/11_CCS/2021_COVID19_analizis/study13_SNMPC_LTV_delta
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. October 12. (2021b)
%

import casadi.*

[~,np] = Epid_Par.GetK;
[s,p,~,p_Cas] = Epid_Par.Symbolic_New;

Cnt(0);
% State variables:
S = sym('S'); J.S = Cnt;
L = sym('L'); J.L = Cnt;
P = sym('P'); J.P = Cnt;
I = sym('I'); J.I = Cnt;
A = sym('A'); J.A = Cnt;
H = sym('H'); J.H = Cnt;
D = sym('D'); J.D = Cnt;
R = sym('R'); J.R = Cnt;
nx = Cnt - 1;
% -----
x = [S;L;P;I;A;H;D;R];

x_Cas = SX(nx,1);
for xi = x.'
    name = char(xi);
    x_Cas(J.(name)) = SX.sym(name);
end

% Osszevont parameterek
% hat_rho = 1/(1/s.tauP + s.pI/s.tauI + (1-s.pI)/s.tauA);
% hat_delta = (1/s.tauP + s.pI/s.tauI + s.qA*(1-s.pI)/s.tauA) * hat_rho;
% hat_eta = s.pI * s.pH;

% Time-dependent parameter: Nr. of vaccinated per day
nu = sym('ImGainRate');
nu_Cas = SX.sym('nu');

% Unknown time-dependent parameters:
beta = sym('TrRate');   % transmission rate
beta_Cas = SX.sym('beta');
omega = sym('ImLossRate');
omega_Cas = SX.sym('w');

Infectious = P + I + s.qA*A + 0.1*H;

dS = -beta*Infectious*S/Np - nu*S + omega*R;
dL = beta*Infectious*S/Np - s.tauL*L;
dP = s.tauL*L - s.tauP*P;
dI = s.pI*s.tauP*P - s.tauI*I;
dA = (1-s.pI)*s.tauP*P - s.tauA*A;
dH = s.tauI*s.pH*I - s.tauH*H;
dD = s.pD*s.tauH*H;
dR = s.tauI*(1-s.pH)*I + s.tauA*A + (1-s.pD)*s.tauH*H + nu*S - omega*R;

f_sym = [dS dL dP dI dA dH dD dR].';
assert(double(simplify(sum(f_sym))) == 0);

Cnt(0);
J.Daily_All = Cnt;
J.Daily_New = Cnt;
J.Rc = Cnt;
J.Rt = Cnt;

h_sym = [
    L + P + I + A + H
    beta * (P + I + s.qA*A) * S / Np
    beta * (1/s.tauP + s.pI/s.tauI + s.qA*(1-s.pI)/s.tauA)
    beta * (1/s.tauP + s.pI/s.tauI + s.qA*(1-s.pI)/s.tauA) * S / Np
    ];

matlabFunction(f_sym,'File','Fn_SLPIAHDR_ode','Vars',{x,p,beta,nu,omega});
matlabFunction(h_sym,'File','Fn_SLPIAHDR_out','Vars',{x,p,beta});

Ts = 1;

f = {};
f.desc = 'f(x,u,p,v)';
f.val = x_Cas + Ts*Fn_SLPIAHDR_ode(x_Cas,p_Cas,beta_Cas,nu_Cas,omega_Cas);
f.Fn = Function('f',{x_Cas,p_Cas,beta_Cas,nu_Cas,omega_Cas},{f.val},{'x','p','beta','nu','omega'},{f.desc});

f.      Input_1 = x;
f.desc__Input_1 = 'State vector (x = [S,L,P,I,A,H,D,R])';
f.      Input_2 = p;
f.desc__Input_2 = 'Parameter vector (p)';
f.      Input_3 = beta;
f.desc__Input_3 = 'Transmission rate of the pathogen (beta)';
f.      Input_4 = nu;
f.desc__Input_4 = 'Immunity gain rate due to vaccination (ImGainRate)';
f.      Input_5 = omega;
f.desc__Input_5 = 'Immunity loss rate due to waning immunity (ImLossRate)';

h = {};
h.desc = '[all_inf,daily_new_inf,R0,Rt]';
h.val = Fn_SLPIAHDR_out(x_Cas,p_Cas,beta_Cas);
h.Fn = Function('h',{x_Cas,p_Cas,beta_Cas},{h.val},{'x','p','beta'},{h.desc});

% h.      Input_1 = x_Cas;
% h.desc__Input_1 = 'State vector (x)';
% h.      Input_2 = beta_Cas;
% h.desc__Input_2 = 'Input vector (u = [beta,w])';
% h.      Input_3 = p_Cas;
% h.desc__Input_3 = 'Parameter vector (p)';

% h.desc__Output_dim1 = 'All infected';
% h.desc__Output_dim2 = 'Daily new infected';
% h.desc__Output_dim3 = 'Basic reproduction number (R0)';
% h.desc__Output_dim4 = 'Time-dependent reproduction number (Rt)';

hvar = {};
hvar.val = zeros(size(h.val));
hvar.Fn = [];
vars = [x_Cas;beta_Cas;p_Cas];
Jh = jacobian(h.val,vars);

[Sigma,Sigma_half] = Pcz_CasADi_Helper.create_sym('Sigma',numel(vars),1,'str','sym');

hvar.val = diag(Jh * Sigma * Jh');
hvar.Fn = Function('hvar',{x_Cas,p_Cas,beta_Cas,Sigma_half},{hvar.val},{'x','p','beta','Sigma'},{'[Var_all_inf,Var_daily_new_inf,Var_Rc]'});

J.nx = numel(x);
J.ny = numel(h.val);
J.np = np;

end
