function [Exp_GP,Var_GP,Cov_xGP,Exp_GP_mean,Var_GP_mean] = GP_Exact_MM_Times(hyp,u,S)
arguments
    hyp struct = struct
    u (:,1) = []
    S (:,:) = []
end
%%
%  File: GP_Exact_MM.m
%  Directory: 5_Sztaki20_Main/Tanulas/11_Exact_Moment_Matching
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. June 18. (2021a)
%  Major review on 2021. July 09. (2021a)
%

persistent K U iK beta Lambda iLambda

Timer_ol3p = pcz_dispFunctionName;

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
X = {hyp.X};
y = {hyp.y};

if isempty(K)
    K = cell(1,nz);
    U = cell(1,nz);
    iK = cell(1,nz);
    beta = cell(1,nz);

    for a = 1:nz
        Timer_o42p = pcz_dispFunctionName(sprintf('Iteration a = %d',a));
        
        [N,n] = size(X{a});

        % Diagonal matrix of the characteristic length values
        Lambda{a} = diag(hyp(a).ell.^2);
        iLambda{a} = diag(1./hyp(a).ell.^2);

        %%% ===================================================================
        % Vectorized matrix values
    
        Timer_kx6p = pcz_dispFunctionName('Vectorize');
    
        % X_3d <-- X(1:N,1,coords)
        X_col__coords_in_3d = permute(X{a},[1 3 2]);
    
        % X_3D' <-- X(1,1:N,coords)
        X_row__coords_in_3d = permute(X{a},[3 1 2]);
    
        % (X_3d - X_3d')(1:N,1:N,coords)
        Delta_XX = X_col__coords_in_3d - X_row__coords_in_3d;
    
        % Value of ``xi - xj'' in [1, Eq. (38)]
        % (X_3d - X_3d')(:,coords) --> (N*N) x n
        Delta_XX_col = reshape(Delta_XX,[N*N n]);
    
        pcz_dispFunctionEnd(Timer_kx6p);
    
        %%% ===================================================================
        % Grammian and the data covariance matrix according to [1, Eq. (3)].
        % Kernel function is given in [1, Eq. (5)].
    
        Timer_l1pq = pcz_dispFunctionName('Exponent_of_K');    
    
        Exponent_of_K = -0.5 * sum((Delta_XX_col * iLambda{a}) .* Delta_XX_col,2);
        K{a} = sf(a)^2 * reshape(exp(Exponent_of_K),[N N]) + sn(a)^2 * eye(N);
    
        pcz_dispFunctionEnd(Timer_l1pq);

        Timer_rxeu = pcz_dispFunctionName('Cholesky decomposition');
        % DIM(1:Na,1:Na)
        U{a} = chol(K{a});
        iK{a} = inv(K{a});
        beta{a} = U{a}\(U{a}'\y{a});
        pcz_dispFunctionEnd(Timer_rxeu);
        
        pcz_dispFunctionEnd(Timer_o42p);
    end

    return
end    


Timer_ru3u = pcz_dispFunctionName('Initialization');
assert(all(eig(S) >= 0),'Input variance not positive (semi-)definite')

% Initialize values
Exp_GP = zeros(nz,1);
Var_GP = zeros(nz,nz);
Cov_xGP = zeros(nw,nz);
Exp_GP_mean = zeros(nz,1);
Var_GP_mean = zeros(nz,1);

Zeta = cell(1,nz);

EXPONENT_of_ka = cell(1,nz);

pcz_dispFunctionEnd(Timer_ru3u);

Timer_ol2p = pcz_dispFunctionName('Loop 1');
for a = 1:nz
    Timer_o42p = pcz_dispFunctionName(sprintf('Iteration a = %d',a));

    Timer_vxzf = pcz_dispFunctionName('Part 1');    

    [~,n] = size(X{a});
    Zeta{a} = X{a} - u';
    
    %%% ===================================================================
    % Covariance of the new input with training samples.
    % Kernel function is given in [1, Eq. (5)].

    Timer_1iqo = pcz_dispFunctionName('EXPONENT_of_ka');    

    EXPONENT_of_ka{a} = -0.5 * sum((Zeta{a} * iLambda{a}) .* Zeta{a},2);
    ka = sf(a)^2 * exp(EXPONENT_of_ka{a});
    
    pcz_dispFunctionEnd(Timer_1iqo);

    pcz_dispFunctionEnd(Timer_vxzf);

    %%% ===================================================================
    % GP regression in u, the expected value of the input [1, Eqs. (8)-(9)]

    Timer_k2fe = pcz_dispFunctionName('Part 2');
    % DIM(1:Na,1)

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

    % DIM(N,1)
    q = sf(a)^2 / sqrt(det(S*iLambda{a} + eye(n))) * exp(EXPONENT_of_33);

    %%% ===================================================================
    % Compute the first moment (mean) 

    % According to [1, Eq. (30)] and [2, Eq. (2.34)].
    % SCALAR
    Exp_GP(a) = beta{a}.' * q;

    %%% ===================================================================
    % Compute the input-output covariance according to [2, Eq. (2.70)].

    % DIM(n,1)
    Cov_xGP(:,a) = sum(S / (S + Lambda{a}) * (Zeta{a}' .* q' .* beta{a}'),2);

    pcz_dispFunctionEnd(Timer_k2fe);    

    pcz_dispFunctionEnd(Timer_o42p);
end
pcz_dispFunctionEnd(Timer_ol2p);

Timer_ofz5 = pcz_dispFunctionName('Loop 2');
for a = 1:nz
    Timer_o42p = pcz_dispFunctionName(sprintf('Iteration a = %d',a));
    [Na,n] = size(X{a});

    % DIM(Na,1,n)
    Za = permute(Zeta{a}*iLambda{a},[1 3 2]);
    
    for b = nz:-1:a
        Timer_8jju = pcz_dispFunctionName(sprintf('Iteration b = %d',b));
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
        pcz_dispFunctionEnd(Timer_8jju);
    end

    % Cross covariance of GP_a and GP_b for input x ~ N(u,S), according
    % to [2, Eq. (2.55)] when a == b
    % Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - trace(K{a}\Q);
    Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - sum(sum(iK{a}.*Q));
    pcz_dispFunctionEnd(Timer_o42p);
end
pcz_dispFunctionEnd(Timer_ofz5);

Timer_dq7u = pcz_dispFunctionName('Post processing');
Var_GP = Var_GP + Var_GP' - diag(diag(Var_GP));

Sigma = [
    S         Cov_xGP
    Cov_xGP.' Var_GP
    ];

if any(eig(Sigma) < 0)
    fprintf('[\b<strong>GP_Exact_MM [javitott] (WARNING):</strong> Joint distribution has an indefinite variance matrix.]\b\n')    
    fprintf('[\bMinimal eigenvalue: %d.]\b\n', min(eig(Sigma)))
end
pcz_dispFunctionEnd(Timer_dq7u);

pcz_dispFunctionEnd(Timer_ol3p);
end

function test
%%

G_reset

n = 10;
N = 1000;

rng(1);
u = randn(n,1);

S = randn(n);
S = S*S' * 1e-5;

hyp = struct();
% --
hyp(1).mean = [];
hyp(1).cov = [ -0.5 , -0.8 , 0.2 , 1.1 , -0.2 , 0.3 , -0.89 , 0.12 , 12.1 , -3.2 , 2.3 ];
hyp(1).lik = -1;
hyp(1).ell = exp(hyp(1).cov(1:end-1));
hyp(1).sf = exp(hyp(1).cov(end));
hyp(1).sn = exp(hyp(1).lik);
hyp(1).X = randn(N,n);
hyp(1).y = randn(N,1);
% --
hyp(2).mean = [];
hyp(2).cov = [ 2.1 , 0.6 , 11.1 , -0.2 , 0.3 , 0.6 , 0.7 , 11.6 , -1.2 , 1.2 , 1.6 ];
hyp(2).lik = -1;
hyp(2).ell = exp(hyp(2).cov(1:end-1));
hyp(2).sf = exp(hyp(2).cov(end));
hyp(2).sn = exp(hyp(2).lik);
hyp(2).X = randn(N,n);
hyp(2).y = randn(N,1);

% --
hyp(3).mean = [];
hyp(3).cov = [ 2.1 , 0.6 , 10.6 , -0.2 , 0.2 , 0.6 , 0.7 , 12.1 , -3.2 , 2.3 , 1.6 ];
hyp(3).lik = -1;
hyp(3).ell = exp(hyp(2).cov(1:end-1));
hyp(3).sf = exp(hyp(2).cov(end));
hyp(3).sn = exp(hyp(2).lik);
hyp(3).X = randn(N,n);
hyp(3).y = randn(N,1);

% Generate persistent variables
GP_Exact_MM_Times(hyp);

[GP_mean,GP_var,GP_cov,GP_mean_ME,GP_var_ME] = GP_Exact_MM_Times(hyp,u,S);

fprintf('Mean equivalent: GP(x) ~ N([%6.3g,%6.3g,%6.3g],[%6.3g,%6.3g,%6.3g])\n', GP_mean_ME, GP_var_ME);
fprintf('Moment matching: GP(x) ~ N([%6.3g,%6.3g,%6.3g],[%6.3g,%6.3g,%6.3g])\n', GP_mean, diag(GP_var));

end
