function K = GP_eval_Kxy_symbolical(hyp,X,z_sym)
%%
%  File: GP_eval_Kxy_symbolical.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. March 29. (2020b)
%

%%
% X     in R(N x n) <-- numerical
% z_sym in R(n x 1) <-- symbolical

%%%
% Dimensions.

[N,~] = size(X);

%%%
% Hyperparameters (numerical values).
% 
% These are given:
% hyp.ell = exp(hyp_tuned.cov(1:end-1));
% hyp.sf  = exp(hyp_tuned.cov(end));
% hyp.sn  = exp(hyp_tuned.lik);         % <--- Csak K_XX-nel kell

iP = diag(1./hyp.ell.^2);
sf = hyp.sf;

if size(X,1) ~= size(z_sym,1)

    %%%
    % Difference.
    
    Delta_Xz = ones(N,1)*z_sym.' - X;

    %%%
    % Inv(P)-weighted norm of each pair of vectors from X and Y.

    Weighted_norm = sum((Delta_Xz * iP) .* Delta_Xz,2);
    
else
   
    %%%
    % Difference and Inv(P)-weighted norm of symbolic vectors X and z_sym.

    Delta_Xz = X-z_sym;
    Weighted_norm = Delta_Xz.' * iP * Delta_Xz;
    
end


%%%
% Covariance matrix.

K = sf^2 * exp(-0.5 * Weighted_norm);

end
