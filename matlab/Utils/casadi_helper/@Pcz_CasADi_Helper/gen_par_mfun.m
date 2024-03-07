function [f_par] = gen_par_mfun(helper,fname,dir,simblk)
arguments
    helper
    fname {mustBeTextScalar}
    dir {mustBeTextScalar} = '.'
    simblk = false
end
%%
%  File: gen_par_mfun.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 25. (2021a)
%  Reviewed on 2021. November 03. (2021b)
%

import casadi.*

[args,argnames] = pcas_struct2args_Type1(helper.P);
f_par = Function('f_par',args,{helper.p},argnames,'p');

pcas_gen_mfun_vectsel(f_par,fname,dir,simblk);

end

function test
%%

import casadi.*

helper = Pcz_CasADi_Helper("SX");

pars = [
    SX.sym('a',4)
    SX.sym('b')
    SX.sym('c',[4,1])
    ];

helper.add_sym('par',pars);

helper.new_par('M',[2,3],4);
helper.new_par('P',3,5,str='sym');
helper.new_par('x',[3,5]);

var_array = helper.new_var('var_array',[2,3]);
var_veccell = helper.new_var('var_veccell',3,4);
var_matcell = helper.new_var('var_matcell',[2,3],2);
var_symcell = helper.new_var('var_symcell',2,3,'str','sym');
var_blksym = helper.new_var('var_blksym',[2,3,4],'str','sym');
var_blksymcell = helper.new_var('var_blksymcell',[1,1,2],2,'str','sym');


helper.gen_var_mfun('fvar_proba',[pwd filesep 'Sandbox']);
helper.gen_par_mfun('fpar_proba',[pwd filesep 'Sandbox']);

edit Sandbox/fpar_proba
edit Sandbox/fvar_proba

% helper.gen_par_mfun('f_par','sim_qarm_lqi_actual/Function proba',1);

end


