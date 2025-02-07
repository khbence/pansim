%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 29. (2023a)
% 
% Decreasing, shrinking, narrowing, shortening horizon. 
% Feedback using the reconstructed, estimated epidemic state.

p_val = [
    0.6667 % tauL
    0.3226 % tauP
    0.2439 % tauA
    0.2500 % tauI
    0.0833 % tauH
    0.7500 % qA
    0.4800 % pI
    0.0760 % pH
    0.4800 % pD
    ];

%% Letrehozok Matlab Symbolic Math Toolbox-fele szimbolikus valtozokat
% (Ezek segitsegevel lehet matlab fuggveny-fajlt generalni.)

% Parameters
tauL = sym('tauL');
tauP = sym('tauP');
tauA = sym('tauA');
tauI = sym('tauI');
tauH = sym('tauH');
qA = sym('qA');
pI = sym('pI');
pH = sym('pH');
pD = sym('pD');
% -----
p = [tauL;tauP;tauA;tauI;tauH;qA;pI;pH;pD];
np = numel(p);

% State variables:
S = sym('S');
L = sym('L');
P = sym('P');
I = sym('I');
A = sym('A');
H = sym('H');
D = sym('D');
R = sym('R');
% -----
x = [S;L;P;I;A;H;D;R];
nx = numel(x);

%% Letrehozok CasADi-fele szimbolikus valtozokat

import casadi.*

x_Cas = SX(nx,1);
for i = 1:nx
    name = char(x(i));
    x_Cas(i) = SX.sym(name);
end

p_Cas = SX(np,1);
for i = 1:np
    name = char(p(i));
    p_Cas(i) = SX.sym(name);
end

% Time-dependent parameter: Nr. of vaccinated per day
nu = sym('nu');
nu_Cas = SX.sym('nu');

% Control input: transmission rate
beta = sym('beta');
beta_Cas = SX.sym('beta');

Infectious = P + I + qA*A + 0.1*H;

Np = 179500; % Population of Hungary
dS = -beta*Infectious*S/Np - nu*S;
dL = beta*Infectious*S/Np - tauL*L;
dP = tauL*L - tauP*P;
dI = pI*tauP*P - tauI*I;
dA = (1-pI)*tauP*P - tauA*A;
dH = tauI*pH*I - tauH*H;
dD = pD*tauH*H;
dR = tauI*(1-pH)*I + tauA*A + (1-pD)*tauH*H + nu*S;

f_sym = [dS;dL;dP;dI;dA;dH;dD;dR];
assert(double(simplify(sum(f_sym))) == 0);

matlabFunction(f_sym,'File','Fn_SLPIAHDR_ode','Vars',{x,p,beta,nu});

Ts = 1;

% State vector (x = [S,L,P,I,A,H,D,R])
% Parameter vector (p)
% Input: Transmission rate of the pathogen (beta)
% Preliminarily known time-dependent disturbance: immunity gain rate due to vaccination
f_Cas = x_Cas + Ts*Fn_SLPIAHDR_ode(x_Cas,p_Cas,beta_Cas,nu_Cas);
f_Fn = Function('f',{x_Cas,p_Cas,beta_Cas,nu_Cas},{f_Cas},{'x','p','beta','nu'},{'f(x,u,p,v)'});

%% Control goal: flatten the curve

FreeT = readtimetable('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-03-09/FreeSpread/FreeSpread_2024-03-14_08-56.xls');

N = 210;
t_sim = 0:N-1;

FreeMean = 72;
FreeStd = 24;
FreePeak = 3300;
Date = t_sim+C.Start_Date;
Ifree = normpdf(t_sim,FreeMean,FreeStd);
Ifree = Ifree / max(Ifree) * FreePeak;

CtrlMean = FreeMean + 7*7;
CtrlStd = 48;
CtrlPeak = 1500;
Iref1 = normpdf(t_sim,CtrlMean,CtrlStd);
Iref1 = Iref1 / max(Iref1) * CtrlPeak;

CtrlMean = FreeMean + 8*7;
CtrlStd = 36;
CtrlPeak = 2500;
Iref3 = normpdf(t_sim,CtrlMean,CtrlStd);
Iref3 = Iref3 / max(Iref3) * CtrlPeak;

CtrlMean = FreeMean - 3*7;
CtrlStd = 20;
CtrlPeak = 750;
Iref41 = normpdf(t_sim,CtrlMean,CtrlStd);
Iref41 = Iref41 / max(Iref41) * CtrlPeak;

CtrlMean = FreeMean + 14*7;
CtrlStd = 30;
CtrlPeak = 1000;
Iref42 = normpdf(t_sim,CtrlMean,CtrlStd);
Iref42 = Iref42 / max(Iref42) * CtrlPeak;

Iref4 = Iref41 + Iref42;
Iref5 = flip(Iref4);

% 2024.03.19. (március 19, kedd), 12:48
CtrlMean = FreeMean + 12*7;
CtrlStd = 48;
CtrlPeak = 1500;
Iref2 = normpdf(t_sim,CtrlMean,CtrlStd);
Iref2 = Iref2 / max(Iref2) * CtrlPeak;
Iref2 = Iref2.^4;
Iref2 = Iref2 / max(Iref2) * 2000;

fig = figure(123); 
delete(fig.Children)
ax = axes(fig);
hold on; grid on; box on;
plot(Date,Iref1,'DisplayName','Scenario 1','LineWidth',2);
plot(Date,Ifree,'DisplayName','Free spread');
plot(Date,Iref2,'DisplayName','Scenario 2','LineWidth',2);
plot(Date,Iref3,'DisplayName','Scenario 3');
plot(Date,Iref4,'DisplayName','Scenario 4','LineWidth',3);
plot(FreeT.Date,FreeT.I,'DisplayName','Free spread');
xlim(Date([1,end]))
ax.YLim(1) = 0;
legend

Iref = Iref1;
Tp = 21;
Tp = 30;

%% Initial guess
% (Vedd eszre, hogy az MPC ismeretlenjeit eleve ugy hozom letre, hogy mar egy potencialis
% erteket is adok neki. Ugy vettem eszre, hogy a kezdeti feltetelen nagyon sok mulik, hogy
% milyen lesz a megoldas, es milyen gyorsan szamolja azt ki.)

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
    A_guess*0 % H
    A_guess*0 % D
    A_guess*0 % R
    ];

% Initial states
x0 = [
    178159
    135
    244
    72
    75
    9
    0
    806
    ];

x_fh = @(x_var) [x0 , x_var];

%%%
%  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
%                         0 0 0 0 0 0 0 , 1 1 1 ... ]
%  Control input can be changed only weekly
Nr_Periods = N / Tp;
idx = reshape(ones(Tp,1) * (1:Nr_Periods),N,1);
I = eye(Nr_Periods);
M = I(:,idx);

beta_guess = ones(1,Nr_Periods) * 0.23;
beta_fh = @(beta_var) beta_var * M;

%% Create optimization problem

helper = Pcz_CasADi_Helper('SX');

x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0);
x = x_fh(x_var);

% Statistics of the estimated transmission rate for the different interventions
% Min             0.1212  
% Median          0.1648  
% Max             0.2301  
% Mean            0.1688  
% Std             0.0208  

beta_min = 0.1212;
beta_max = 0.2301;
beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);
beta = beta_fh(beta_var);

% Enforce the state equations
for i = 1:N
    x_kp1 = f_Fn(x(:,i),p_val,beta(i),0);
    helper.add_eq_con( x_kp1 - x(:,i+1) );
end

% Minimize the tracking error
helper.add_obj('I_error',(x_var(4,:) - Iref).^2,0.001);

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;
sol_guess = full(Fn_var(x_guess,beta_guess));   

% Solve the control optimization problem
ret = NL_solver.solve([],sol_guess);

% Get the solution
beta_sol = helper.get_value('beta');
beta = beta_fh(beta_sol);
x_sol = helper.get_value('x');
x = x_fh(x_sol);

fig = figure(1);
delete(fig.Children)
nexttile;
plot(0:N,x(4,:),1:N,Iref), title tracking
nexttile;
plot(1:N,beta), title input
