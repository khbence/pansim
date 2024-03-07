function K = GP_eval_Kxy_numerical(hyp,X,Z)
%%
%  File: GP_eval_Kxy_numerical.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. March 29. (2020b)
%
% Based on function `covSEard'
%
% """
% Wrapper for Squared Exponential covariance function covSE.m.
%
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x,z) = sf^2 * exp(-(x-z)'*inv(P)*(x-z)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-04-27.
%
% See also covSE.m.
% """
% 
%       [x1_1, x2_1, ...., xn_1]
%   X = [x1_2, x2_2, ...., xn_2] in R(N x n)
%       [...., ...., ...., ....]
%       [x1_N, x2_N, ...., xn_N]
%
%       [z1_1, z2_1, ...., zn_1]
%   Z = [z1_2, z2_2, ...., zn_2] in R(M x n)
%       [...., ...., ...., ....]
%       [z1_M, z2_M, ...., zn_M]
%
%                     [xk_1 - zk_1, .... , xk_1 - zk_M]
%   Delta_XZ(:,:,k) = [..........., .... , ...........]
%                     [xk_N - zk_1, .... , xk_N - zk_M]
%   ahol k = 1,...,n.
%

%%%
% Dimensions.

[N,~] = size(X);
[M,n] = size(Z);

%%%
% Hyperparameters.
% 
% These are given:
% hyp.ell = exp(hyp_tuned.cov(1:end-1));
% hyp.sf  = exp(hyp_tuned.cov(end));
% hyp.sn  = exp(hyp_tuned.lik);         % <--- Csak K_XX-nel kell

iP = diag(1./hyp.ell.^2);
sf = hyp.sf;

%%%
% Difference with a few matrix manipulations.

% X_3d <-- X(1:N,1,coords)
X_col__coords_in_3d = permute(X,[1 3 2]);

% Y_3D' <-- Y(1,1:M,coords)
Z_row__coords_in_3d = permute(Z,[3 1 2]);

% (X_3d - Y_3d')(1:N,1:M,coords)
Delta_XZ = X_col__coords_in_3d - Z_row__coords_in_3d;

% (X_3d - Y_3d')(:,coords) --> (N*M) x n
Delta_XZ_col = reshape(Delta_XZ,[N*M n]);

%%%
% Inv(P)-weighted norm of each pair of vectors from X and Y.

Weighted_norm_col = sum((Delta_XZ_col * iP) .* Delta_XZ_col,2);
Weighted_norm = reshape(Weighted_norm_col,[N M]);

%%%
% Covariance matrix.

K = sf^2 * exp(-0.5 * Weighted_norm);

end


function test1_trivial
%%

hyp = [ -0.5 , 0.3 ];

x = [
    1
    2
    3
    4
    ];


K = GP_eval_Kxy_numerical(hyp,x,x)

K_ref = covSEard(hyp,x)

K_ref - K

exp(2*0.3)

end