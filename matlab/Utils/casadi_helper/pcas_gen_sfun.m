function pcas_gen_sfun(f,fname,dir)
%%
%  File: pcz_gen_sfun.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 25. (2021a)
%

arguments
    f {mustBeA(f,'casadi.Function')}
    fname {mustBeTextScalar} = '';
    dir {mustBeTextScalar} = '';
end

if isempty(dir)
    dir = pwd;
end

if isempty(fname)
    fname = f.name;
end

import casadi.*

actual_dir = pwd;
cd(dir)

cg_sfun_opts = struct;
cg_sfun_opts.verbose = true;
cg_sfun_opts.indent = 4;
cg_sfun_opts.casadi_real = 'real_T';
cg_sfun_opts.real_min    = 'real_T'; % Needed if you code-generate sqpmethod method
cg_sfun_opts.casadi_int  = 'int_T';
cg_sfun_opts.with_header = true;

cg_mex_opts = cg_sfun_opts;
cg_mex_opts.main = false;
cg_mex_opts.mex = true;

% Generate mex
mexname = [fname '_mex'];
cg = CodeGenerator(mexname,cg_mex_opts);
cg.add_include('simstruc.h');
cg.add(f);
cg.generate();
mex([mexname '.c'])

[sname,~,cname] = pcas_gen_sfun_c(dir,fname,f.name);
cg = CodeGenerator(fname,cg_sfun_opts);
cg.add_include('simstruc.h');
cg.add(f);
cg.generate();
mex(sname,cname);

cd(actual_dir);

end