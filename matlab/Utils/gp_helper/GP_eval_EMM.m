function [Exp_GP,Var_GP,Cov_xGP] = GP_eval_EMM(hyp,u,S,args) %#codegen
%%
%  File: GP_Exact_MM.m
%  Directory: 5_Sztaki20_Main/Tanulas/11_Exact_Moment_Matching
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. June 18. (2021a)
%  Major review on 2021. July 09. (2021a)
%

persistent K U iK beta Lambda iLambda

if nargin == 0 && nargout == 0
    test
    return
end

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

nz = numel(hyp);
nw = numel(hyp(1).ell);

sn = [hyp.sn];
sf = [hyp.sf];
X = {hyp.X};w
y = {hyp.y};

if isempty(K) || nargin < 3
    K = cell(1,nz);
    U = cell(1,nz);
    iK = cell(1,nz);
    beta = cell(1,nz);

    for a = 1:nz        
        [N,n] = size(X{a});

        % Diagonal matrix of the characteristic length values
        Lambda{a} = diag(hyp(a).ell.^2);
        iLambda{a} = diag(1./hyp(a).ell.^2);

        %%% ===================================================================
        % Vectorized matrix values
    
        % X_3d <-- X(1:N,1,coords)
        X_col__coords_in_3d = permute(X{a},[1 3 2]);
    
        % X_3D' <-- X(1,1:N,coords)
        X_row__coords_in_3d = permute(X{a},[3 1 2]);
    
        % (X_3d - X_3d')(1:N,1:N,coords)
        Delta_XX = X_col__coords_in_3d - X_row__coords_in_3d;
    
        % Value of ``xi - xj'' in [1, Eq. (38)]
        % (X_3d - X_3d')(:,coords) --> (N*N) x n
        Delta_XX_col = reshape(Delta_XX,[N*N n]);
    
        %%% ===================================================================
        % Grammian and the data covariance matrix according to [1, Eq. (3)].
        % Kernel function is given in [1, Eq. (5)].
    
        Exponent_of_K = -0.5 * sum((Delta_XX_col * iLambda{a}) .* Delta_XX_col,2);
        K{a} = sf(a)^2 * reshape(exp(Exponent_of_K),[N N]) + sn(a)^2 * eye(N);
    
        % DIM(1:Na,1:Na)
        U{a} = chol(K{a});
        iK{a} = inv(K{a});
        beta{a} = U{a}\(U{a}'\y{a});
    end

    return
end    


assert(all(eig(S) >= 0),'Input variance not positive (semi-)definite')

% Initialize values
Exp_GP = zeros(nz,1);
Var_GP = zeros(nz,nz);
Cov_xGP = zeros(nw,nz);
% Exp_GP_mean = zeros(nz,1);
% Var_GP_mean = zeros(nz,1);

Zeta = cell(1,nz);
q = cell(1,nz);

EXPONENT_of_ka = cell(1,nz);

for a = 1:nz
    [~,n] = size(X{a});
    Zeta{a} = X{a} - u';
    
    %%% ===================================================================
    % Covariance of the new input with training samples.
    % Kernel function is given in [1, Eq. (5)].

    EXPONENT_of_ka{a} = -0.5 * sum((Zeta{a} * iLambda{a}) .* Zeta{a},2);

    %%% ===================================================================
    % Compute the values of lj according to [1, Eq. (33)]
    % The same (as `qj') appears in [2, Eqs. (2.36) and (2.63)].

    % DIM(N,1)
    EXPONENT_of_33 = -0.5 * sum((Zeta{a} / (S + Lambda{a})) .* Zeta{a},2);

    % DIM(N,1)
    q{a} = sf(a)^2 / sqrt(det(S*iLambda{a} + eye(n))) * exp(EXPONENT_of_33);

    %%% ===================================================================
    % Compute the first moment (mean) 

    % According to [1, Eq. (30)] and [2, Eq. (2.34)].
    % SCALAR
    Exp_GP(a) = beta{a}.' * q{a};
end

if args.MeanOnly
    return
end

for a = 1:nz
    %%% ===================================================================
    % Compute the input-output covariance according to [2, Eq. (2.70)].

    % DIM(n,1)
    Cov_xGP(:,a) = sum(S / (S + Lambda{a}) * (Zeta{a}' .* q{a}' .* beta{a}'),2);
end

for a = 1:nz
    [Na,n] = size(X{a});

    % DIM(Na,1,n)
    Za = permute(Zeta{a}*iLambda{a},[1 3 2]);
    
    for b = nz:-1:a
        [Nb,~] = size(X{b});
        
        % DIM(1,Nb,n)
        Zb = permute(Zeta{b}*iLambda{b},[3 1 2]);

        % Values of z_{ij} in [2] betweem Eqs. (2.52) and (2.53).
        Z = Za + Zb;                  % DIM(Na,Nb,n)
        Z_col = reshape(Z,[Na*Nb n]); % DIM(Na*Nb,n)
        
        % Value of R in [2] betweem Eqs. (2.52) and (2.53)
        % DIM(n,n)
        R = S*(iLambda{a} + iLambda{b}) + eye(n);
        
        % Third quadratic term of the exponent in [2, Eq. (2.54)].
        % This is SLOW:
        QUADR_TERM_3_of_2_54_col = 0.5 * sum((Z_col / R * S) .* Z_col,2);  % DIM(Na*Nb,1)
        QUADR_TERM_3_of_2_54 = reshape(QUADR_TERM_3_of_2_54_col,[Na,Nb]);  % DIM(Na,Nb)
        
        % Values of n_{ij}^2 in [2, Eq. (2.54)]
        EXPONENT_of_2_54 = 2*(log(sf(a)) + log(sf(b))) ...
            + EXPONENT_of_ka{a} + EXPONENT_of_ka{b}' ...
            + QUADR_TERM_3_of_2_54;
        
        % Final value of matrix Q in [2, Eq. (2.53)]
        % DIM(Na,Nb)
        Q = exp(EXPONENT_of_2_54) / sqrt(det(R));
        
        % Cross covariance of GP_a and GP_b for input x ~ N(u,S), according
        % to [2, Eq. (2.55)] when a ~= b
        Var_GP(a,b) = beta{a}' * Q * beta{b} - Exp_GP(a)*Exp_GP(b);
    end

    % Cross covariance of GP_a and GP_b for input x ~ N(u,S), according
    % to [2, Eq. (2.55)] when a == b
    % Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - trace(K{a}\Q);
    Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - sum(sum(iK{a}.*Q));
end

Var_GP = Var_GP + Var_GP' - diag(diag(Var_GP));

end

%{

codegen -report GP_eval_EMM.m -args {hyp,u,S,args}
% https://www.mathworks.com/help/coder/gs/generating-mex-functions-from-matlab-code-at-the-command-line.html

%}

function code_generation
%%


n = 5;
N = 100;

rng(1);
u = randn(n,1);

S = randn(n);
S = S*S' * 1e-5;

hyp = struct();
% --
hyp(1).mean = [];
hyp(1).cov = [ -0.5 , -0.8 , 0.2 , 1.1 , -0.2 , 0.3 ];
hyp(1).lik = -1;
hyp(1).ell = exp(hyp(1).cov(1:end-1));
hyp(1).sf = exp(hyp(1).cov(end));
hyp(1).sn = exp(hyp(1).lik);
hyp(1).X = randn(N,n);
hyp(1).y = randn(N,1);
% --
hyp(2).mean = [];
hyp(2).cov = [ 2.1 , 0.6 , 10.6 , -0.2 , 0.2 , 0.6 ];
hyp(2).lik = -1;
hyp(2).ell = exp(hyp(2).cov(1:end-1));
hyp(2).sf = exp(hyp(2).cov(end));
hyp(2).sn = exp(hyp(2).lik);
hyp(2).X = randn(N,n);
hyp(2).y = randn(N,1);

% codegen -report GP_eval_EMM_init.m -args {hyp}

tic
[K,U,iK,beta,Lambda,iLambda] = GP_eval_EMM_init_mex(hyp);
toc

tic
[K,U,iK,beta,Lambda,iLambda] = GP_eval_EMM_init(hyp);
toc

% codegen -report GP_eval_EMM_impl.m -args {hyp,u,S,K,U,iK,beta,Lambda,iLambda}

tic
[Exp_GP,Var_GP,Cov_xGP] = GP_eval_EMM_impl_mex(hyp,u,S,K,U,iK,beta,Lambda,iLambda);
toc

tic
[Exp_GP,Var_GP,Cov_xGP] = GP_eval_EMM_impl(hyp,u,S,K,U,iK,beta,Lambda,iLambda);
toc

end
