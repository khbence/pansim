function [GP_mean,GP_var] = GP_eval(hyp,Z,GP_eval_Kxy)
arguments
    hyp struct
    Z = []
    GP_eval_Kxy = @GP_eval_Kxy_numerical
end
%%
%  File: GP_eval_Peti.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. January 20. (2020b)
%

persistent L alpha

%%%
% Dimensions.

Nr_GP = numel(hyp);
Dim_GP = numel(hyp(1).ell);

if isempty(Z)
    %%%
    % Offline computations

    L = cell(1,Nr_GP);
    alpha = cell(1,Nr_GP);

    for a = 1:Nr_GP
    
        % sf = hyp(a).sf;
        sn = hyp(a).sn;
        X = hyp(a).X;
        y = hyp(a).y;
    
    
        %%%
        % Szamolasok.
    
        % Kovariancia matrixok:
        K_XX = GP_eval_Kxy_numerical(hyp(a),X,X) + sn^2 * eye(size(X,1));
    
        % Segedvaltozok:
        L{a} = chol(K_XX);
        alpha{a} = L{a}\(L{a}'\y);
        
    end
    %%% 
    % Log likelihood:
    % GP_loglik = -0.5*y'*alpha{a} - sum(log(diag(L{a}))) - N/2*log(2*pi);

    GP_mean = hyp;
    return
end


if size(Z,2) ~= Dim_GP
    Z = Z.';
end

Dim_out = size(Z,1);

GP_mean = zeros(Dim_out,Nr_GP);
GP_var = zeros(Dim_out,Nr_GP);

for a = 1:Nr_GP

    %%%
    % Hyperparameters.

    % Adottak:
    % hyp.ell = exp(hyp_tuned.cov(1:end-1));
    % hyp.sf  = exp(hyp_tuned.cov(end));
    % hyp.sn  = exp(hyp_tuned.lik);

    sf = hyp(a).sf;
    sn = hyp(a).sn;
    X = hyp(a).X;
    % y = hyp(a).y;

    % Kovariancia matrixok:
    K_XZ = GP_eval_Kxy(hyp(a),X,Z);

    % Segedvaltozok:
    v = L{a}'\K_XZ;

    % A lenyeg
    GP_mean(:,a) = K_XZ' * alpha{a};
    GP_var(:,a) = sf^2 + sn^2 - sum(v.*v)';

    %%% 
    % Log likelihood:
    % GP_loglik = -0.5*y'*alpha{a} - sum(log(diag(L{a}))) - size(X,1)/2*log(2*pi);

end

end
