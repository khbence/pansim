function [y_mean,y_var] = GP_eval_Cas(Z,hyp)
%%
%  File: GP_eval_Cas.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. January 20. (2020b)
%

import casadi.*

%%%
% Hyperparameters.

Nr_GP = numel(hyp);
Dim_GP = numel(hyp(1).ell);

% Adottak:
% hyp.ell = exp(hyp_tuned.cov(1:end-1));
% hyp.sf  = exp(hyp_tuned.cov(end));
% hyp.sn  = exp(hyp_tuned.lik);         % <--- Csak K_XX-nel nem nulla



%%%
% Dimensions.

if size(Z,2) ~= Dim_GP
    Z = Z.';
end

%%%
% Casadi object specific variables.

if isa(Z,'casadi.MX')
    Cas = 'MX';
    I = @(N) MX.eye(N);
    y_mean = MX.zeros(Nr_GP,1);
    y_var = MX.zeros(Nr_GP,1);
elseif isa(Z,'casadi.SX')
    Cas = 'SX';
    I = @(N) SX.eye(N);
    y_mean = SX.zeros(Nr_GP,1);
    y_var = SX.zeros(Nr_GP,1);
elseif isnumeric(Z)
    % Utolag lett kiegeszitve:
    % 2021.04.21. (április 21, szerda), 20:43
    I = @(N) eye(N);
    y_mean = zeros(Nr_GP,1);
    y_var = zeros(Nr_GP,1);
end

for a = 1:Nr_GP

    iP = diag(1./hyp(a).ell);
    sf = hyp(a).sf;
    sn = hyp(a).sn;
    X = hyp(a).X;
    y = hyp(a).y;


    %%%
    % Szamolasok.

    % Kovariancia matrixok:
    K_XX = GP_eval_Kxy_Cas(hyp(a),X,X) + sn^2 * I(size(X,1));
    K_XZ = GP_eval_Kxy_Cas(hyp(a),X,Z);

    % Segedvaltozok:
    L = chol(K_XX);
    alpha = L\(L'\y);
    v = L'\K_XZ;

    % A lenyeg
    y_mean(a) = K_XZ' * alpha;
    y_var(a) = (sf^2 + sn^2 - sum(v.*v))';

    %%% 
    % Erre most nincs szukseg:
    % Likelihood = -0.5*y_train'*alpha - sum(log(diag(R))) - N/2*log(2*pi);

end

end
