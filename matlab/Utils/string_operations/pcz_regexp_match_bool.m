function [ret] = pcz_regexp_match_bool(str, expr, varargin)
%% 
%  
%  file:   pcz_regexp_match_bool.m
%  author: Polcz Péter <ppolcz@gmail.com> 
%  
%  Created on 2017.02.14. Tuesday, 23:51:41
%

opts.mode = 'any'; % any|all
opts = parsepropval(opts, varargin{:});

nrexcl = numel(expr);
nrvars = numel(str);

blacklist = cell(nrexcl, nrvars);
for i = 1:nrexcl
    blacklist(i,:) = regexp(str, expr{i})';
end

if strcmp(opts.mode, 'any')
    ret = sum(~cellfun(@isempty,blacklist),1);
elseif strcmp(opts.mode, 'all')
    ret = prod(~cellfun(@isempty,blacklist),1);
else
    error('`mode` should be `any` or `all`!')
end

end