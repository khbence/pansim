function [ret,symvars] = pcas_symvar(A)
%%
%  File: pcas_symvar.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. November 17. (2021b)
%

symvars = A.symvar;
[ret,I] = sort(cellfun(@(sv) {sv.name}, symvars));
ret = ret.';
symvars = vertcat(symvars{I});

% strjoin(ret,', ');

end