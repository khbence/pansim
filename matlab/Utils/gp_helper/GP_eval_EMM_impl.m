function [Exp_GP,Var_GP,Cov_xGP] = GP_eval_EMM_impl(hyp,u,S,iK,beta,Lambda,iLambda) %#codegen
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
nw = numel(hyp(1).ell);

% sn = [hyp.sn];
sf = [hyp.sf];
X = {hyp.X};
% y = {hyp.y};

% Initialize values
Exp_GP = zeros(nz,1);
Var_GP = zeros(nz,nz);
Cov_xGP = zeros(nw,nz);

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

% if args.MeanOnly
%     return
% end

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

        if a == b
            % Cross covariance of GP_a and GP_b for input x ~ N(u,S), according
            % to [2, Eq. (2.55)] when a == b
            % Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - trace(K{a}\Q);
            Var_GP(a,a) = Var_GP(a,a) + sf(a)^2 - sum(sum(iK{a}.*Q));
        end
    end
end

Var_GP = Var_GP + Var_GP' - diag(diag(Var_GP));

end

%{

codegen -report GP_eval_EMM.m -args {hyp,u,S,args}
% https://www.mathworks.com/help/coder/gs/generating-mex-functions-from-matlab-code-at-the-command-line.html

%}