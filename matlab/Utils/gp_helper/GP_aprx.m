function [ret1,ret2] = GP_aprx(hyp,Xs,args)
arguments
    hyp struct
    Xs = []
    args.Approx {mustBeMember(args.Approx,['FITC','SPEP','VFE'])} = 'FITC' 
end
%%
%  File: FITC_eval.m
%  Directory: 5_Sztaki20_Main/Tanulas/13_SparseGP_Methods
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2022. March 02. (2021b)
%

persistent Luu R alpha Approx

%%%
% Dimensions.

Nr_GP = numel(hyp);
Dim_GP = numel(hyp(1).ell);
N = size(hyp(1).X,1);
M = size(hyp(1).Xu,1);

if isempty(Xs) || ~strcmp(Approx,args.Approx)
    %%%
    % Offline computations

    Luu = cell(1,Nr_GP);
    R = cell(1,Nr_GP);
    alpha = cell(1,Nr_GP);
    Approx = args.Approx;

    hyp_sparse = rmfield(hyp,'Xu');

    for a = 1:Nr_GP
    
        sf = hyp(a).sf;
        sn = hyp(a).sn;
        X = hyp(a).X;
        y = hyp(a).y;
        Xu = hyp(a).Xu;

        % Covariance matrices
        Kuu = GP_eval_Kxy_numerical(hyp,Xu,Xu);
        Kuf = GP_eval_Kxy_numerical(hyp,Xu,X);   Kfu = Kuf';
        Kff = GP_eval_Kxy_numerical(hyp,X,X);
        
        % Stabilize Kuu
        [S,D] = eig(Kuu);
        d = diag(D);
        d(d < 0) = 0;
        D = diag(d);
        Kuu = S * D / S;
        Luu{a} = chol(Kuu);
        
        Af = Luu{a}'\Kuf;
        switch args.Approx
            case { 'FITC', 'SPEP' }
                Qffd = diag(sum(Af.*Af,1));
                Kffd = diag(diag(Kff));
                Lambda = Kffd - Qffd + sn^2 * eye(N);
            case 'VFE'
                Lambda = sn^2 * eye(size(X,1));
        end
        iLambda = diag(1./diag(Lambda));
        
        iSigma = Kuu + Kuf * iLambda * Kfu;
        R{a} = chol(iSigma);
        
        B = R{a}' \ Kuf;
        beta = B*iLambda*y;
        alpha{a} = R{a} \ beta;

        % Evaluate GP approximation for the original data set:
        Qss = sum(Af.*Af,1)';
        GP_var = sf^2 + sn^2 - Qss + sum(B.*B,1)';

        % Sparse pseudo inputs and outputs:
        hyp_sparse(a).X = Xu;
        hyp_sparse(a).y = (Kuu + hyp_sparse(a).sn^2 * eye(M)) / R{a} * beta;
        GP_eval(hyp_sparse);

        [~,GP_var_SP] = GP_eval(hyp_sparse,X);

        dst_init = mean(sqrt(GP_var) - sqrt(GP_var_SP));
        dst = dst_init;
        It = 0;

        while abs(dst) > 0.001 && It < 10
            hyp_sparse(a).sn = hyp_sparse(a).sn + dst;
            hyp_sparse(a).y = (Kuu + hyp_sparse(a).sn^2 * eye(M)) / R{a} * beta;
            GP_eval(hyp_sparse);
            
            [~,GP_var_SP] = GP_eval(hyp_sparse,X);
    
            dst = mean(sqrt(GP_var) - sqrt(GP_var_SP));
        end

        % if abs(dst) > abs(dst_init)
        %     warning('Initial distance: %d, iterated distance: %d', dst, dst_init)
        %     Seged_Mat = inv(inv(Kuu) - inv(iSigma)) - Kuu;
        %     Eig_Seged = eig(Seged_Mat);
        %     hyp_sparse(a).sn = sqrt(max(Eig_Seged));
        %     
        %     dst = GP_dist(hyp(a),hyp_sparse(a))
        %     if abs(dst) > abs(dst_init)
        %         warning('Initial distance: %d, achieved distance (max-eig): %d', dst, dst_init)
        %         hyp_sparse(a).sn = hyp(a).sn;
        %         hyp_sparse(a).y = (Kuu + hyp_sparse(a).sn^2 * eye(M)) / R{a} * beta;
        %     end
        % end
    end

    if isempty(Xs)
        ret1 = hyp_sparse;
        return
    end
end


if size(Xs,2) ~= Dim_GP
    Xs = Xs.';
end

Dim_out = size(Xs,1);

GP_mean = zeros(Dim_out,Nr_GP);
GP_var = zeros(Dim_out,Nr_GP);

for a = 1:Nr_GP

    sf = hyp(a).sf;
    sn = hyp(a).sn;
    Xu = hyp(a).Xu;

    % Covariance matrices
    Kus = GP_eval_Kxy_numerical(hyp,Xu,Xs);
    
    As = Luu{a}'\Kus;
    Qss = sum(As.*As,1)';
    
    B = R{a}'\Kus;
    
    GP_mean(:,a) = Kus' * alpha{a};
    GP_var(:,a) = sf^2 + sn^2 - Qss + sum(B.*B,1)';

end

ret1 = GP_mean;
ret2 = GP_var;

end