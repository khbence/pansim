function [f,F] = get_obj(o,x,p,s)
arguments
    o
    x = []
    p = []
    s.Overwrite = true
end
%%
%  File: get_obj.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. November 03. (2021b)
%

import casadi.*

if isempty(x)
    x = o.x_sol;
elseif s.Overwrite
    o.x_sol = x;
end

if isempty(p)
    p = o.p_val;
elseif s.Overwrite
    o.p_val = p;
end

F = o.F;
f = pcas_full(Function('Fun_f',{o.x,o.p},{o.f},{'x','p'},{'f'}),x,p);

if nargout > 1

    fns = fieldnames(F);
    for i = 1:numel(fns)
        s = F.(fns{i});

        ss = struct();
        ss.J = full(s.f_J(x,p));
        ss.Jk = full(s.f_Jk(x,p));
        ss.w = pcas_full(Function('f_w',{o.p},{s.w}),p);
        
        F.(fns{i}) = ss;
        F.([fns{i} '_SUM']) = ss.J;
    end

    F.f = f;
end
    
end