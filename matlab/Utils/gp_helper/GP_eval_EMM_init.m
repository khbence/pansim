function [K,U,iK,beta,Lambda,iLambda] = GP_eval_EMM_init(hyp) %#codegen
%%
%  File: GP_Exact_MM.m
%  Directory: 5_Sztaki20_Main/Tanulas/11_Exact_Moment_Matching
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. June 18. (2021a)
%  Major review on 2021. July 09. (2021a)
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

nz = numel(hyp);

sn = [hyp.sn];
sf = [hyp.sf];
X = {hyp.X};
y = {hyp.y};

K = cell(1,nz);
U = cell(1,nz);
iK = cell(1,nz);
beta = cell(1,nz);
Lambda = cell(1,nz);
iLambda = cell(1,nz);

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


end

function test
%%

    n = 12;
    N = 300;
    
    rng(1);
    X = randn(N,n);
    y = randn(N,1);
        u = randn(n,1);
    S = randn(n);
    S = S*S' * 1e-3;
    
    hyp = struct();
    hyp.mean = [];
    hyp.cov = [ -0.5 , -0.8 , -0.2 , -0.6 , -0.5 , -0.7 , -0.5 , -0.8 , -0.54 , -0.9 , -0.5 , -0.8 , 0.3 ];
    hyp.lik = -1;
    hyp.ell = exp(hyp.cov(1:end-1));
    hyp.sf = exp(hyp.cov(end));
    hyp.sn = exp(hyp.lik);
    hyp.X = X;
    hyp.y = y;
    
    cdr_config = coder.config;
    cdr_config.EnableAutoParallelization = true;
    cdr_config.NumberOfCpuThreads = 4;
    cdr_config.OptimizeReductions = true;

    [K,U,iK,beta,Lambda,iLambda] = GP_eval_EMM_init(hyp);
    codegen -report GP_eval_EMM_impl.m -args {hyp,u,S,iK,beta,Lambda,iLambda} -config cdr_config

    Nr_Comp = 10;

    u = randn(n,1);
    S = randn(n);
    S = S*S' * 1e-3;

    tic
    for i = 1:Nr_Comp
        [Exp_GP,Var_GP,Cov_xGP] = GP_eval_EMM_impl(hyp,u + sin(i),S + eye(n)*(1+sin(i)),iK,beta,Lambda,iLambda);
    end
    toc

    tic
    for i = 1:Nr_Comp
        [Exp_GP,Var_GP,Cov_xGP] = GP_eval_EMM_impl_mex(hyp,u + sin(i),S + eye(n)*(1+sin(i)),iK,beta,Lambda,iLambda);
    end
    toc

end

%{

codegen -report GP_eval_EMM.m -args {hyp,u,S,args}
% https://www.mathworks.com/help/coder/gs/generating-mex-functions-from-matlab-code-at-the-command-line.html

%}