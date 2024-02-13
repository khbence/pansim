function [y_mean,y_var] = GP_eval(Z,hyp,GP_eval_Kxy)
%%
%  File: GP_eval_Peti.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. January 20. (2020b)
%

arguments
    Z
    hyp
    GP_eval_Kxy = @GP_eval_Kxy_numerical
end

Nr_GP = numel(hyp);
Dim_GP = numel(hyp(1).ell);

sZ = size(Z);
if sZ(2) ~= Dim_GP
    Z = Z.';
end

Dim_out = size(Z,1);

y_mean = zeros(Dim_out,Nr_GP);
y_var = zeros(Dim_out,Nr_GP);

for a = 1:Nr_GP

    %%%
    % Hyperparameters.

    % Adottak:
    % hyp.ell = exp(hyp_tuned.cov(1:end-1));
    % hyp.sf  = exp(hyp_tuned.cov(end));
    % hyp.sn  = exp(hyp_tuned.lik);         % <--- Csak K_XX-nel nem nulla

    iP = diag(1./hyp(a).ell);
    sf = hyp(a).sf;
    sn = hyp(a).sn;
    X = hyp(a).X;
    y = hyp(a).y;

    % Kovariancia matrixok:
    K_XX = GP_eval_Kxy_numerical(hyp(a),X,X) + sn^2 * eye(size(X,1));
    K_XZ = GP_eval_Kxy(hyp(a),X,Z);

    % Segedvaltozok:
    L = chol(K_XX);
    alpha = L\(L'\y);
    v = L'\K_XZ;

    % A lenyeg
    y_mean(:,a) = K_XZ' * alpha;
    y_var(:,a) = (sf^2 + sn^2 - sum(v.*v))';

end

end
