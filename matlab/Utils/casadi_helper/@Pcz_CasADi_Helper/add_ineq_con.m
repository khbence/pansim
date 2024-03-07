function add_ineq_con(o,conname,fun,s)
%%
%  File: add_ineq_con.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 29. (2021a)
%

arguments
    o

    conname {mustBeValidVariableName}
    
    fun (:,1)
    
    s.lb (:,1) {mustBeNumeric} = -Inf
    s.ub (:,1) {mustBeNumeric} = Inf

end

if isscalar(s.ub)
    s.ub = zeros(size(fun)) + s.ub;
end

if isscalar(s.lb)
    s.lb = zeros(size(fun)) + s.lb;
end        

if ~isfield(o.G,conname)
    o.G.(conname) = struct(g = [], lb = [], ub = []);
end

con = o.G.(conname);

o.G.(conname).g = [ con.g fun ];
o.G.(conname).lb = [ con.lb s.lb ];
o.G.(conname).ub = [ con.ub s.ub ];

end