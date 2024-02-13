function [args,argnames] = pcas_struct2args_Type1(s)
%%
%  File: pcas_struct2args_Type1.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. November 03. (2021b)
%


fns = fieldnames(s);

n = numel(fns);

s = cellfun(@(fn) {s.(fn)}, fns);
s_idx = cellfun(@(s) {s.idx},s);
s_half = cellfun(@(s) {s.half},s);
s_numel = cellfun(@(h) numel(h),s_half);
s_iscell = cellfun(@(h) iscell(h),s_half);

s_argnr = s_numel .* s_iscell + double(~s_iscell);

s_arg_names = cell(1,n);
for i = 1:n
    if s_argnr(i) == 1
        s_arg_names{i} = fns(i);
    else
        s_arg_names{i} = cellfun(@(j) {[fns{i} num2str(j)]},num2cell(0:s_argnr(i)-1));
    end    
    
    if ~s_iscell(i)
        s_half{i} = s_half(i);
    end
end

args = [s_half{:}];
argnames = [s_arg_names{:}];

end