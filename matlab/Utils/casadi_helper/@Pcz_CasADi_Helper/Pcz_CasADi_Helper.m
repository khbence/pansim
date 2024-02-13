classdef Pcz_CasADi_Helper < handle
%%
%  File: Pcz_CasADi_Helper.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 09. (2020b)
%

properties

X = struct
P = struct
F = struct
G = struct
% H = struct

%--------------------------------------------------------------------------
x     % Variables
x_sol % Solution
lbx   % Upper and lower bounds:
ubx   % lbx(p) <= x <= ubx(p)

p     % Parameter
p_val % Sample values of the parameters

f = 0 % Cost function: f(x,p)

g     % Inequality constraints:
lbg   % lbg(p) <= g(x,p) <= ubg(p)
ubg

h     % Equality constraints: h(x,p) == 0
%--------------------------------------------------------------------------

CasX = "SX"

end


methods

    function o = Pcz_CasADi_Helper(CasX)
        if nargin > 0
            o.CasX = CasX;
        end
    end
           
    % ---------------------------------------------------------------------

    [varargout] = add_sym(varargin)
    [varargout] = new_sym(varargin)
    
    function [str,name,var] = new_var(o,varargin)
        [str,name,var] = o.new_sym("var",varargin{:});
    end    
    
    function [str,name,var] = new_par(o,varargin)
        [str,name,var] = o.new_sym("par",varargin{:});
    end
    
    % ---------------------------------------------------------------------

    add_ineq_con(varargin)
    add_obj(o,objname,J,w)
    
    function add_eq_con(o,fun)
        arguments
            o,fun (:,1)
        end
        o.h = [ o.h ; fun ];
    end
    
    [varargout] = get_qp_solver(varargin)
    
    [varargout] = get_nl_solver(varargin)

    [varargout] = gen_par_mfun(varargin)
    [varargout] = gen_var_mfun(varargin)

    [varargout] = construct(varargin)    

    [varargout] = get_obj(varargin)
    [varargout] = get_value(varargin)
    
end % methods

methods (Static)
    
    [varargout] = resolve_lb_ub__(varargin)
    
    [varargout] = create_sym(varargin)
    
end % static methods

end
