function [f_var] = gen_var_mfun(helper,fname,dir,simblk)
arguments
    helper
    fname {mustBeTextScalar} = ''
    dir {mustBeTextScalar} = '.'
    simblk = false
end
%%
%  File: gen_var_mfun.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. November 03. (2021b)
%  
%

import casadi.*

[args,argnames] = pcas_struct2args_Type1(helper.X);
f_var = Function('f_var',args,{helper.x},argnames,'vars');

if ~isempty(fname)
    pcas_gen_mfun_vectsel(f_var,fname,dir,simblk);
end

end


