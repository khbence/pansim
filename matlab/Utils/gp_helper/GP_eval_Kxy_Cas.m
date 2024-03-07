function K = GP_eval_Kxy_Cas(hyp,X,Z)
%%
%  File: GP_eval_Kxy_Cas.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. March 29. (2020b)
%

import casadi.*

%%
% X     in R(N x n) <-- numerical
% z_sym in R(n x 1) <-- symbolical
% 

%%%
% Dimensions.

[Nx,nx] = size(X);
[Nz,nz] = size(Z);

assert(nx == nz);

%%%
% Hyperparameters (numerical values).
% 
% These are given:
% hyp.ell = exp(hyp_tuned.cov(1:end-1));
% hyp.sf  = exp(hyp_tuned.cov(end));
% hyp.sn  = exp(hyp_tuned.lik);         % <--- Csak K_XX-nel kell

iP = diag(1./hyp.ell.^2);
sf = hyp.sf;

Weighted_norm = SX(Nx,Nz);
for i = 1:Nz

    %%%
    % Difference.
    
    Delta_Xz = ones(Nx,1)*Z(i,:) - X;

    %%%
    % Inv(P)-weighted norm of each pair of vectors from X and Y.

    Weighted_norm(:,i) = sum((Delta_Xz * iP) .* Delta_Xz,2);
        
end


%%%
% Covariance matrix.

K = sf^2 * exp(-0.5 * Weighted_norm);

end

function test
%%

import casadi.*

X = SX.sym('X',3,2)
Z = SX.sym('Z',4,2)

hyp.ell = SX.sym('l',2)
hyp.sf = SX.sym('sf',1)

K = GP_eval_Kxy_Cas(hyp,X,Z)

end

