function [Var_full,varname,var] = new_sym(o,type,varname,dims,N,s)
%%
%  File: new_sym.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 19. (2020b)
%

arguments
    o
    type {mustBeMember(type,["var","par"])}
    varname {mustBeValidVariableName}
    dims (1,:) {mustBeNumeric}
    N (1,1) {mustBeNumeric} = 1
    
    s.str {mustBeMember(s.str,[
        "full","sym"
        ])} = "full"
    
    s.lb (:,1) = -Inf
    s.ub (:,1) = Inf
    s.val (:,1) {mustBeNumeric} = 0
end

[Var_full,Var_half,var,f_str,f_vec,defval] = Pcz_CasADi_Helper.create_sym(...
    varname,dims,N,...
    str = s.str,...
    CasX = "SX",...
    r1 = "Var_full",...
    r2 = "Var_half",...
    r3 = "var",...
    r4 = "f_str",...
    r5 = "f_vec",...
    r6 = "defval");

s = rmfield(s,'str');

if isscalar(s.val) && s.val == 0
    s.val = defval;
end

o.add_sym(type,Var_full,var,...
    half = Var_half,...
    f_str = f_str,...
    f_vec = f_vec,...
    name = varname,...
    val = s.val, lb = s.lb, ub = s.ub);

end