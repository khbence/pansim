function hyp = GP_hyp(hyp_tuned,X,y)
%%
%  File: GP_hyp.m
%  Directory: 5_Sztaki20_Main/Tanulas/11_Exact_Moment_Matching
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. June 18. (2021a)
%

if ~iscell(X)
    X = {X};
    y = {y};
end

hyp = repmat(struct,[1,numel(hyp_tuned)]);
for a = 1:numel(hyp_tuned)
    hyp(a).legacy = "Legacy form of hyperparameters (inherited from GPML):";
    hyp(a).mean = hyp_tuned(a).mean;
    hyp(a).cov = hyp_tuned(a).cov;
    hyp(a).lik = hyp_tuned(a).lik;
    hyp(a).newinfo = "New form of hyperparameters:";
    hyp(a).ell = exp(hyp_tuned(a).cov(1:end-1));
    hyp(a).sf = exp(hyp_tuned(a).cov(end));
    hyp(a).sn = exp(hyp_tuned(a).lik);
    hyp(a).X = X{a};
    hyp(a).y = y{a};

    if isfield(hyp_tuned,'xu')
        hyp(a).Xu = hyp_tuned(a).xu;
    end
end

end
