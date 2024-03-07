classdef Pcz_Abstract_Solver < handle
%%
%  File: Pcz_Abstract_Solver.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com)
%
%  Created on 2021. April 19. (2020b)
%

properties

helper
userdata = struct

end

methods

    function o = Pcz_Abstract_Solver(helper,varargin)
    %%
        o.helper = helper;
        o.helper.x_sol = zeros(size(o.helper.x))*nan;

    end

    function set_param(o,varname,var)
    %%
        if ~ischar(varname)
            if nargin < 3
                var = varname;
            end
            varname = inputname(2);            
        end
        
        if ~iscell(var)
            var = {var};
        end
        
        s = o.helper.P.(varname);
        o.helper.p_val(s.idx) = full(s.f_vec(var{:}));
        
        o.helper.P.(varname).set = true;
        
    end
    
    function set_params(o,p_val)
    %%
        o.helper.p_val = p_val;
        
    end
    
    function reuse_param(o,varargin)
    %%
        for i = 1:numel(varargin)
            if ~ischar(varargin{i})
                varargin{i} = inputname(i+1);
            end
            o.helper.P.(varargin{i}).set = true;
        end
        
    end
    
    function var = get_value(o,varname,varargin)
    %%
        warning 'use Pcz_CasADi_Helper function instead'

        if ~ischar(varname)
            varname = inputname(2);
        end
        s = o.helper.X.(varname);
        
        type = 'full';
        if numel(varargin) > 1
            type = varargin{1};
        end
        
        var = cell(1,s.f_str.n_out);
        [var{:}] = s.f_str(o.helper.x_sol(s.idx));

        switch type
            case 'full'
                var = cellfun(@(v) {full(v)},var);
            case 'sparse'
                var = cellfun(@(v) {sparse(v)},var);
            case 'DM'
            otherwise
                error('type should be `full|sparse|DM`')
        end
        
        if numel(var) == 1
            var = var{1};
        end

    end
    
    function [f,F,F_fh] = get_obj(o)
    %%
        warning 'use Pcz_CasADi_Helper function instead'

        import casadi.*
        
        F = o.helper.F;
        Fun_f = Function('Fun_f',{o.helper.x,o.helper.p},{o.helper.f},{'x','p'},{'f'});
        F_fh = @(x,p) full(Fun_f(x,p));
        f = full(Fun_f(o.helper.x_sol,o.helper.p_val));

        if nargout > 1
    
            fns = fieldnames(F);
            for i = 1:numel(fns)
                s = F.(fns{i});
    
                ss = struct();
                ss.J_fh = @(x,p) full(s.f_J(x,p));
                ss.J = full(s.f_J(o.helper.x_sol,o.helper.p_val));
                ss.Jk = full(s.f_Jk(o.helper.x_sol,o.helper.p_val));
                
                f_w = Function('f_w',{o.helper.p},{s.w});
                ss.w = full(f_w(o.helper.p_val));
                
                F.(fns{i}) = ss;
                F.([fns{i} '_SUM']) = ss.J;
            end

        end
        
    end
    
    function check_params(o)
    %%
        s = o.helper.P;
        fns = fieldnames(s);
        for i = 1:numel(fns)
            fn = fns{i};

            if ~s.(fn).set
                warning('Parameter `%s` is not setted. Using default value',fn)
            end
        end

    end
    
end

end
