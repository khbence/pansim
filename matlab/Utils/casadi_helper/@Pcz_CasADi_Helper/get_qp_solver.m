function [QP_solver] = get_qp_solver(o)
%%
%  File: get_qp_solver.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 19. (2020b)
%

import casadi.*

% Finalize helper object
o.construct;

% Generate a symbolic sparse zero vector
switch o.CasX
    case 'SX'
        zero_x = SX(zeros(size(o.x)));        
    case 'MX'
        zero_x = MX(zeros(size(o.x)));
end


%%%
% [1] Objective function: f(x,p) = 0.5 x' H(p) x + x' f(p) + const(p)

% Create CasADi function for the objective function
fx_f = Function('fx_f',{o.x},{o.f});

% Gradient of the cost function
df = gradient(o.f,o.x);
fx_df = Function('fx_df',{o.x},{df});

% quadprog:arg1, H: Hession of the cost function
H = hessian(o.f,o.x);

% quadprog:arg2, f:
f = fx_df(zero_x);

% const, the constant term (not required for quadprog)
const = fx_f(zero_x);


%%%
% [2] Inequality constraints (not implemented yet)

A = [];
b = [];

%%%
% [3] Equality constraints: h(x,p) == 0 --> B x == c

fx_h = Function('fx_h',{o.x},{o.h});

% quadprog:arg5, B:
B = jacobian(o.h,o.x);

% quadprog:arg6, c:
c = -fx_h(zero_x);

%%%
% Ellenorzeskeppen
%{

    f_ZERO = Function('f_ZERO',{o.x,o.p},{o.h - B*o.x + c});
    full(f_ZERO(zeros(size(o.x)),zeros(size(o.p))))

%}

l = o.lbx;
u = o.ubx;

f_QP_matrices = Function('f_QP_matrices',{o.p},{H,f,A,b,B,c,l,u});

QP_solver = Pcz_QP_Solver(o,f_QP_matrices);

end