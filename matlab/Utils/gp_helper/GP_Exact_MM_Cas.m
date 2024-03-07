function [Exp_GP,Var_GP,Cov_xGP,Exp_GP_mean,Var_GP_mean] = GP_Exact_MM_Cas(hyp,u,S)
arguments
    hyp struct
    u (:,1)
    S (:,:)
end
%%
%  File: GP_Exact_MM_Cas.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v6_2021_10_01/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
% 
%  Created on 2021. June 18. (2021a)
%  Modified on 2021. September 30. (2021b)
%

%% 
% The implementation is based on:
% 
% [1] Candela, Girard and Rasmussen (2003). Prediction at an uncertain
% input for Gaussian processes and relevance vector machines application to
% multiple-step ahead time-series forecasting. Tech. Report IMM-2003-18,
% Technical University of Denmark, 2003.
% 
% [2] Mark Peter Deisenroth. Efficient reinforcement learning using
% gaussian processes -- Revised version. Faculty of Informatics Institute
% for Anthropomatics Intelligent Sensor-Actuator-Systems Laboratory (ISAS),
% 2017.

import casadi.*

nz = numel(hyp);
nw = numel(hyp(1).ell);

n1 = round(nw/2);

% Initialize values
Exp_GP = SX(nz,1);
Var_GP = SX(nz,nz);
Cov_xGP = SX(nw,nz);
Exp_GP_mean = SX(nz,1);
Var_GP_mean = SX(nz,1);

K = cell(1,nz);
U = cell(1,nz);
Ki = cell(1,nz);

Lambda = cell(1,nz);
iLambda = cell(1,nz);

Zeta = cell(1,nz);
beta = cell(1,nz);

EXPONENT_of_ka = cell(1,nz);

sn = [hyp.sn];
sf = [hyp.sf];
X = {hyp.X};
y = {hyp.y};

for a = 1:nz
    [N,nw] = size(X{a});
    
    % Diagonal matrix of the characteristic length values
    Lambda{a} = diag(hyp(a).ell.^2);
    iLambda{a} = diag(1./hyp(a).ell.^2);
    Zeta{a} = X{a} - ones(N,1) * u';

    %%% ===================================================================
    % Vectorized matrix values

    % This may vary by a, as the number of samples in each GP may vary
    Index_Matrix = ones(N,1) * (1:N);
    Index_MatrixT = Index_Matrix'; 
    Index_1 = Index_MatrixT(:);
    Index_2 = Index_Matrix(:);

    % X_3d <-- X(1:N,1,coords)
    X_col__coords_in_3d = X{a}(Index_1,:);

    % X_3d' <-- X(1,1:N,coords)
    X_row__coords_in_3d = X{a}(Index_2,:);

    % (X_3d - X_3d')(1:N,1:N,coords)
    Delta_XX = X_col__coords_in_3d - X_row__coords_in_3d;

    %%% ===================================================================
    % Grammian and the data covariance matrix according to [1, Eq. (3)].
    % Kernel function is given in [1, Eq. (5)].

    Exponent_of_K = -0.5 * sum((Delta_XX * iLambda{a}) .* Delta_XX,2);
    K{a} = sf(a)^2 * reshape(exp(Exponent_of_K),[N N]) + sn(a)^2 * eye(N);

    %%% ===================================================================
    % Covariance of the new input with training samples.
    % Kernel function is given in [1, Eq. (5)].

    EXPONENT_of_ka{a} = -0.5 * sum((Zeta{a} * iLambda{a}) .* Zeta{a},2);
    ka = sf(a)^2 * exp(EXPONENT_of_ka{a});

    %%% ===================================================================
    % GP regression in u, the expected value of the input [1, Eqs. (8)-(9)]

    % DIM(1:Na,1:Na)
    U{a} = chol(K{a});
    Ki{a} = U{a}\(U{a}'\eye(N));

    % DIM(1:Na,1)
    beta{a} = U{a}\(U{a}'\y{a});

    % DIM(1:Na,1)
    v = U{a}'\ka;

    % SCALAR
    Exp_GP_mean(a) = beta{a}.' * ka;

    % SCALAR
    Var_GP_mean(a) = sf(a)^2 - v.'*v;

    % Erdekes, hogy valahol mashol igy szamolom, de miert?
    % y_var = sf^2 + sn^2 - sum(v.*v);

    %%% ===================================================================
    % Compute the values of lj according to [1, Eq. (33)]
    % The same (as `qj') appears in [2, Eqs. (2.36) and (2.63)].

    % DIM(N,1)
    EXPONENT_of_33 = -0.5 * sum((Zeta{a} / (S + Lambda{a})) .* Zeta{a},2);

    M = S + Lambda{a};
    if nw > 1
        A = M(1:n1,1:n1);
        B = M(1:n1,n1+1:nw);
        C = M(n1+1:nw,1:n1);
        D = M(n1+1:nw,n1+1:nw);
        Determinant = det(A) * det(D - C/A*B) / det(Lambda{a});
    else
        Determinant = det(M) / det(Lambda{a});
    end

    % DIM(N,1)
    q = sf(a)^2 / sqrt(Determinant) * exp(EXPONENT_of_33);

    %%% ===================================================================
    % Compute the first moment (mean) 

    % According to [1, Eq. (30)] and [2, Eq. (2.34)].
    % SCALAR
    Exp_GP(a) = beta{a}.' * q;

    %%% ===================================================================
    % Compute the input-output covariance according to [2, Eq. (2.70)].

    % DIM(n,1)
    Cov_xGP(:,a) = sum(S / (S + Lambda{a}) * (ones(nw,1) * ( q' .* beta{a}' ) .* Zeta{a}'),2);

end
    
for a = 1:nz
    [Na,nw] = size(X{a});

    % DIM(Na,n)
    Za = Zeta{a}*iLambda{a};
    
    for b = nz:-1:a
        [Nb,~] = size(X{b});
        
        % This may vary by a, as the number of samples in each GP may vary
        Index_Matrix_a = (ones(Nb,1) * (1:Na))';
        Index_Matrix_b = (ones(Na,1) * (1:Nb)); 
        Index_a = Index_Matrix_a(:);
        Index_b = Index_Matrix_b(:);
        
        % DIM(1,Nb,n)
        Zb = Zeta{b}*iLambda{b};

        % Values of z_{ij} in [2] between Eqs. (2.52) and (2.53).
        Z_col = Za(Index_a,:) + Zb(Index_b,:); % DIM(Na*Nb,n)
        
        % Value of R in [2] betweem Eqs. (2.52) and (2.53)
        % DIM(n,n)
        R = S*(iLambda{a} + iLambda{b}) + eye(nw);
    
        M = S + inv(iLambda{a} + iLambda{b});
        if nw > 1
            A = M(1:n1,1:n1);
            B = M(1:n1,n1+1:nw);
            C = M(n1+1:nw,1:n1);
            D = M(n1+1:nw,n1+1:nw);
            Determinant_of_R = det(A) * det(D - C/A*B) * det(iLambda{a} + iLambda{b});
        else
            Determinant_of_R = det(M) * det(iLambda{a} + iLambda{b});
        end

        % Third quadratic term of the exponent in [2, Eq. (2.54)].
        QUADR_TERM_3_of_2_54_col = 0.5 * sum((Z_col / R * S) .* Z_col,2);  % DIM(Na*Nb,1) [SLOW]
        QUADR_TERM_3_of_2_54 = reshape(QUADR_TERM_3_of_2_54_col,[Na,Nb]);  % DIM(Na,Nb)

        % Values of n_{ij}^2 in [2, Eq. (2.54)]
        EXPONENT_of_2_54 = 2*(log(sf(a)) + log(sf(b))) ...
            + EXPONENT_of_ka{a}*ones(1,Nb) + ones(Na,1)*EXPONENT_of_ka{b}' ...
            + QUADR_TERM_3_of_2_54;
        
        % Final value of matrix Q in [2, Eq. (2.53)]
        % DIM(Na,Nb)
        Q = exp(EXPONENT_of_2_54) / sqrt(Determinant_of_R);
        
        % Cross covariance of GP_a and GP_b for input x ~ N(u,S), according
        % to [2, Eq. (2.55)] when a ~= b
        Var_GP(a,b) = beta{a}' * Q * beta{b} - Exp_GP(a)*Exp_GP(b);
    end
    
    % Cross covariance of GP_a and GP_b for input x ~ N(u,S), according
    % to [2, Eq. (2.55)] when a == b
    % Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - trace(K{a}\Q); % <-- slow
    Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - sum(sum(Ki{a}.*Q));
end

Var_GP = Var_GP + Var_GP' - diag(diag(Var_GP));

Sigma = [
    S         Cov_xGP
    Cov_xGP.' Var_GP
    ];

end

function test1
%%

import casadi.*
CasX = "SX";

helper = Pcz_CasADi_Helper(CasX);

n = 2;
N = 5;

rng(1);
X = randn(N,n);
y = randn(N,1);

u = helper.create_sym('u',[n,1],"CasX",CasX);
[S,S_half] = helper.create_sym('S',n,"str","sym","r2","Var_half");

u_val = randn(n,1);
S_val = randn(n);
S_val = S_val*S_val' * 1e-3;

hyp = struct();
hyp.mean = [];
hyp.cov = [ -0.5 , -0.8 , 0.3 ];
hyp.lik = -1;
hyp.ell = exp(hyp.cov(1:end-1));
hyp.sf = exp(hyp.cov(end));
hyp.sn = exp(hyp.lik);
hyp.X = X;
hyp.y = y;

[GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME] = GP_Exact_MM_Cas(hyp,u,S);
[GP_mean_val1,GP_var_val1,GP_cov_val1,GP_mean_ME_val1,GP_var_ME_val1] = ...
    GP_Exact_MM(hyp,u_val,S_val);

[GP_mean_val2,GP_var_val2,GP_cov_val2,GP_mean_ME_val2,GP_var_ME_val2] = ...
    pcas_full(Function('f',{u,S_half},{GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME}),u_val,S_val);

fprintf('(Num) Mean equivalent: GP(x) ~ N(%6.3g,%6.3g)\n',GP_mean_ME_val1,GP_var_ME_val1);
fprintf('(Cas) Mean equivalent: GP(x) ~ N(%6.3g,%6.3g)\n',GP_mean_ME_val2,GP_var_ME_val2);
fprintf('(Num) Moment matching: GP(x) ~ N(%6.3g,%6.3g), Cov(x,GP(x)) = (%6.3g,%6.3g)\n',GP_mean_val1,GP_var_val1,GP_cov_val1);
fprintf('(Cas) Moment matching: GP(x) ~ N(%6.3g,%6.3g), Cov(x,GP(x)) = (%6.3g,%6.3g)\n',GP_mean_val2,GP_var_val2,GP_cov_val2);

assert(norm( ...
    [GP_mean_val1,GP_var_val1,GP_cov_val1',GP_mean_ME_val1,GP_var_ME_val1] ...
    - ...
    [GP_mean_val2,GP_var_val2,GP_cov_val2',GP_mean_ME_val2,GP_var_ME_val2] ...
    ) < 1e-10);

end

function test2
%%

import casadi.*
CasX = "SX";

helper = Pcz_CasADi_Helper(CasX);

n = 5;
N = 10;

rng(1);
X = randn(N,n);
y = randn(N,1);

u = helper.create_sym('u',[n,1],"CasX",CasX);
[S,S_half] = helper.create_sym('S',n,"str","sym","r2","Var_half");

u_val = randn(n,1);
S_val = randn(n);
S_val = S_val*S_val' * 1e-3;

hyp = struct();
hyp.mean = [];
hyp.cov = [ -0.5 , -0.8 , 0.2 , 1.1 , -0.2 , 0.3 ];
hyp.lik = -1;
hyp.ell = exp(hyp.cov(1:end-1));
hyp.sf = exp(hyp.cov(end));
hyp.sn = exp(hyp.lik);
hyp.X = X;
hyp.y = y;

[GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME] = GP_Exact_MM_Cas(hyp,u,S);
[GP_mean_val1,GP_var_val1,GP_cov_val1,GP_mean_ME_val1,GP_var_ME_val1] = ...
    GP_Exact_MM(hyp,u_val,S_val);

[GP_mean_val2,GP_var_val2,GP_cov_val2,GP_mean_ME_val2,GP_var_ME_val2] = ...
    pcas_full(Function('f',{u,S_half},{GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME}),u_val,S_val);

fprintf('(Num) Mean equivalent: GP(x) ~ N(%6.3g,%6.3g)\n',GP_mean_ME_val1,GP_var_ME_val1);
fprintf('(Cas) Mean equivalent: GP(x) ~ N(%6.3g,%6.3g)\n',GP_mean_ME_val2,GP_var_ME_val2);
fprintf('(Num) Moment matching: GP(x) ~ N(%6.3g,%6.3g), Cov(x,GP(x)) = (%6.3g,%6.3g,%6.3g,%6.3g,%6.3g)\n',GP_mean_val1,GP_var_val1,GP_cov_val1);
fprintf('(Cas) Moment matching: GP(x) ~ N(%6.3g,%6.3g), Cov(x,GP(x)) = (%6.3g,%6.3g,%6.3g,%6.3g,%6.3g)\n',GP_mean_val2,GP_var_val2,GP_cov_val2);

assert(norm( ...
    [GP_mean_val1,GP_var_val1,GP_cov_val1',GP_mean_ME_val1,GP_var_ME_val1] ...
    - ...
    [GP_mean_val2,GP_var_val2,GP_cov_val2',GP_mean_ME_val2,GP_var_ME_val2] ...
    ) < 1e-10);

end

function test3
%%

import casadi.*
CasX = "SX";

helper = Pcz_CasADi_Helper(CasX);

n = 2;
N1 = 3;
N2 = 4;

u = helper.create_sym('u',[n,1],"CasX",CasX);
[S,S_half] = helper.create_sym('S',n,"str","sym","r2","Var_half");

u_val = randn(n,1);
S_val = randn(n);
S_val = S_val*S_val' * 1e-3;

hyp = struct();
% --
hyp(1).mean = [];
hyp(1).cov = [ -0.5 , -0.8 , 0.3 ];
hyp(1).lik = -1;
hyp(1).ell = exp(hyp(1).cov(1:end-1));
hyp(1).sf = exp(hyp(1).cov(end));
hyp(1).sn = exp(hyp(1).lik);
hyp(1).X = randn(N1,n);
hyp(1).y = randn(N1,1);
% --
hyp(2).mean = [];
hyp(2).cov = [ 2.1 , 0.6 , 0.6 ];
hyp(2).lik = -1;
hyp(2).ell = exp(hyp(2).cov(1:end-1));
hyp(2).sf = exp(hyp(2).cov(end));
hyp(2).sn = exp(hyp(2).lik);
hyp(2).X = randn(N2,n);
hyp(2).y = randn(N2,1);

[GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME] = GP_Exact_MM_Cas(hyp,u,S);
[GP_mean_val1,GP_var_val1,GP_cov_val1,GP_mean_ME_val1,GP_var_ME_val1] = ...
    GP_Exact_MM(hyp,u_val,S_val);

[GP_mean_val2,GP_var_val2,GP_cov_val2,GP_mean_ME_val2,GP_var_ME_val2] = ...
    pcas_full(Function('f',{u,S_half},{GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME}),u_val,S_val);

fprintf('(Num) Mean equivalent: GP(x) ~ N([%6.3g,%6.3g],diag([%6.3g,%6.3g]))\n',GP_mean_ME_val1,GP_var_ME_val1);
fprintf('(Cas) Mean equivalent: GP(x) ~ N([%6.3g,%6.3g],diag([%6.3g,%6.3g]))\n',GP_mean_ME_val2,GP_var_ME_val2);
fprintf('(Num) Moment matching: GP(x) ~ N([%6.3g,%6.3g],diag([%6.3g,%6.3g;%6.3g,%6.3g])), Cov(x,GP(x)) = [%6.3g,%6.3g;%6.3g,%6.3g]\n',GP_mean_val1,GP_var_val1,GP_cov_val1');
fprintf('(Cas) Moment matching: GP(x) ~ N([%6.3g,%6.3g],diag([%6.3g,%6.3g;%6.3g,%6.3g])), Cov(x,GP(x)) = [%6.3g,%6.3g;%6.3g,%6.3g]\n',GP_mean_val2,GP_var_val2,GP_cov_val2');

assert(norm( ...
    [GP_mean_val1,GP_var_val1,GP_cov_val1',GP_mean_ME_val1,GP_var_ME_val1] ...
    - ...
    [GP_mean_val2,GP_var_val2,GP_cov_val2',GP_mean_ME_val2,GP_var_ME_val2] ...
    ) < 1e-10);

end


function test4_big
%%

import casadi.*
CasX = "SX";
rng(1)

helper = Pcz_CasADi_Helper(CasX);

n = 12;
N1 = 300;
N2 = 200;
N3 = 100;

u = helper.create_sym('u',[n,1],"CasX",CasX);
[S,S_half] = helper.create_sym('S',n,"str","sym","r2","Var_half");

u_val = randn(n,1);
S_val = randn(n);
S_val = S_val*S_val' * 1e-3;

hyp = struct();
% --
hyp(1).mean = [];
hyp(1).cov = randn(1,n+1);
hyp(1).lik = -1;
hyp(1).ell = exp(hyp(1).cov(1:end-1));
hyp(1).sf = exp(hyp(1).cov(end));
hyp(1).sn = exp(hyp(1).lik);
hyp(1).X = randn(N1,n);
hyp(1).y = randn(N1,1);
% --
hyp(2).mean = [];
hyp(2).cov = randn(1,n+1);
hyp(2).lik = -1;
hyp(2).ell = exp(hyp(2).cov(1:end-1));
hyp(2).sf = exp(hyp(2).cov(end));
hyp(2).sn = exp(hyp(2).lik);
hyp(2).X = randn(N2,n);
hyp(2).y = randn(N2,1);

% --
hyp(3).mean = [];
hyp(3).cov = randn(1,n+1);
hyp(3).lik = -1;
hyp(3).ell = exp(hyp(3).cov(1:end-1));
hyp(3).sf = exp(hyp(3).cov(end));
hyp(3).sn = exp(hyp(3).lik);
hyp(3).X = randn(N3,n);
hyp(3).y = randn(N3,1);

[GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME] = GP_Exact_MM_Cas(hyp,u,S);
[GP_mean_val1,GP_var_val1,GP_cov_val1,GP_mean_ME_val1,GP_var_ME_val1] = ...
    GP_Exact_MM(hyp,u_val,S_val);

[GP_mean_val2,GP_var_val2,GP_cov_val2,GP_mean_ME_val2,GP_var_ME_val2] = ...
    pcas_full(Function('f',{u,S_half},{GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME}),u_val,S_val);

Error = norm( ...
    [GP_mean_val1,GP_var_val1,GP_cov_val1',GP_mean_ME_val1,GP_var_ME_val1] ...
    - ...
    [GP_mean_val2,GP_var_val2,GP_cov_val2',GP_mean_ME_val2,GP_var_ME_val2] ...
    );
assert(Error < 1e-10,'Approximation error higher then the tolerance: %d > 1e-10',Error);

end

function test_determinant
%%
n = 12;

L = diag(n+1:2*n);
I = eye(n);

[S,S_half] = Pcz_CasADi_Helper.create_sym('S',n,"str","sym","r1","Var_half");

n1 = round(n/2);
n2 = n - n1;

M = S*L + I;

A = M(1:n1,1:n1);
B = M(1:n1,n1+1:n);
C = M(n1+1:n,1:n1);
D = M(n1+1:n,n1+1:n);

tic; det(A) * det(D - C/A*B); toc

% tic; det(S + L); toc
% 
% tic; det(S*L + I); toc
% 
% tic; det(S*L); toc
% 
% tic; det(S); toc

end

function test_determinant_num
%%

n = 12;
n1 = round(n/2);

L = diag(randn(1,n).^2*5+0.5);

S = randn(12);
S = S*S';

I = eye(n);

[ trace(I - inv(S*L + I)) , log(det(S*L + I)) , trace(S*L) ]

M = S*L + I;

A = M(1:n1,1:n1);
B = M(1:n1,n1+1:n);
C = M(n1+1:n,1:n1);
D = M(n1+1:n,n1+1:n);

det_M1 = det(A) * det(D - C/A*B)
det_M2 = det(M)

RelativeError = (det_M1 - det_M2) / det_M1

end
