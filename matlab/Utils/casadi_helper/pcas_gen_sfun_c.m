function [sname,hname,cname] = pcas_gen_sfun_c(dir,fname,Casfname)
%%
%  File: pcz_gen_sfun_c.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 25. (2021a)
%


sname = [ fname '_sfun' ];
hname = [ fname '.h' ];
cname = [ fname '.c' ];

sfun_code = strjoin({
    sprintf(...
    '#define S_FUNCTION_NAME %s',sname)
    '#define S_FUNCTION_LEVEL 2'
    '#include "simstruc.h"'
    sprintf(...
    '#include "%s"',hname)
    'static void mdlInitializeSizes(SimStruct *S)'
    '{'
    '    ssSetNumSFcnParams(S, 0);'
    '    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {'
    '        return; /* Parameter mismatch will be reported by Simulink */'
    '    }'
    '    /* Read in CasADi function dimensions */'
    sprintf(...
    '    int_T n_in  = %s_n_in();',Casfname)
    sprintf(...
    '    int_T n_out = %s_n_out();',Casfname)
    '    int_T sz_arg, sz_res, sz_iw, sz_w;'
    sprintf(...
    '    %s_work(&sz_arg, &sz_res, &sz_iw, &sz_w);',Casfname)
    '    /* Set up simulink input/output ports */'
    '    int_T i;'
    '    if (!ssSetNumInputPorts(S, n_in)) return;'
    '    for (i=0;i<n_in;++i) {'
    sprintf(...
    '      const int_T* sp = %s_sparsity_in(i);',Casfname)
    '      /* Dense inputs assumed here */'
    '      ssSetInputPortDirectFeedThrough(S, i, 1);'
    '      ssSetInputPortMatrixDimensions(S, i, sp[0], sp[1]);'
    '    }'
    '    if (!ssSetNumOutputPorts(S, n_out)) return;'
    '    for (i=0;i<n_out;++i) {'
    sprintf(...
    '      const int_T* sp = %s_sparsity_out(i);',Casfname)
    '      /* Dense outputs assumed here */'
    '      ssSetOutputPortMatrixDimensions(S, i, sp[0], sp[1]);'
    '    }'
    '    ssSetNumSampleTimes(S, 1);'
    '    /* Set up CasADi function work vector sizes */'
    '    ssSetNumRWork(S, sz_w);'
    '    ssSetNumIWork(S, sz_iw);'
    '    ssSetNumPWork(S, sz_arg+sz_res);'
    '    ssSetNumNonsampledZCs(S, 0);'
    '    /* specify the sim state compliance to be same as a built-in block */'
    '    ssSetSimStateCompliance(S, USE_DEFAULT_SIM_STATE);'
    '    ssSetOptions(S,'
    '                 SS_OPTION_WORKS_WITH_CODE_REUSE |'
    '                 SS_OPTION_EXCEPTION_FREE_CODE |'
    '                 SS_OPTION_USE_TLC_WITH_ACCELERATOR);'
    '    /* Signal that we want to use the CasADi Function */'
    sprintf(...
    '    %s_incref();',Casfname)
    '}'
    '/* Function: mdlInitializeSampleTimes ========================================='
    ' * Abstract:'
    ' *    Specifiy that we inherit our sample time from the driving block.'
    ' */'
    'static void mdlInitializeSampleTimes(SimStruct *S)'
    '{'
    '    ssSetSampleTime(S, 0, INHERITED_SAMPLE_TIME);'
    '    ssSetOffsetTime(S, 0, 0.0);'
    '    ssSetModelReferenceSampleTimeDefaultInheritance(S); '
    '}'
    'static void mdlOutputs(SimStruct *S, int_T tid)'
    '{'
    '    /* Read in CasADi function dimensions */'
    sprintf(...
    '    int_T n_in  = %s_n_in();',Casfname)
    sprintf(...
    '    int_T n_out = %s_n_out();',Casfname)
    '    int_T sz_arg, sz_res, sz_iw, sz_w;'
    sprintf(...
    '    %s_work(&sz_arg, &sz_res, &sz_iw, &sz_w);',Casfname)
    '    /* Set up CasADi function work vectors */'
    '    void** p = ssGetPWork(S);'
    '    const real_T** arg = (const real_T**) p;'
    '    p += sz_arg;'
    '    real_T** res = (real_T**) p;'
    '    real_T* w = ssGetRWork(S);'
    '    int_T* iw = ssGetIWork(S);'
    '    /* Point to input and output buffers */'
    '    int_T i;   '
    '    for (i=0; i<n_in;++i) {'
    '      arg[i] = *ssGetInputPortRealSignalPtrs(S,i);'
    '    }'
    '    for (i=0; i<n_out;++i) {'
    '      res[i] = ssGetOutputPortRealSignal(S,i);'
    '    }'
    '    /* Get a hold on a location to read/write persistant internal memory'
    '    */'
    sprintf(...
    '    int mem = %s_checkout();',Casfname)
    '    /* Run the CasADi function */'
    sprintf(...
    '    %s(arg,res,iw,w,mem);',Casfname)
    '    /* Release hold */'
    sprintf(...
    '    %s_release(mem);',Casfname)
    '}'
    'static void mdlTerminate(SimStruct *S) {'
    '  /* Signal that we no longer want to use the CasADi Function */'
    sprintf(...
    '  %s_decref();',Casfname)
    '}'
    '#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */'
    '#include "simulink.c"      /* MEX-file interface mechanism */'
    '#else'
    '#include "cg_sfun.h"       /* Code generation registration function */'
    '#endif'
    },newline);

sname = [sname '.c'];

fid = fopen([dir filesep sname],'wt');
fprintf(fid,'%s',sfun_code);
fclose(fid);

end
