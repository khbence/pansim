function add_obj(o, objname, J, w)
%%
%  File: add_obj.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 29. (2021a)
%

J = J(:);

if ~isfield(o.F,objname)
    o.F.(objname) = struct('J',[],'w',[]);        
end

if nargin < 4
    if isempty(o.F.(objname).w)
        w = 1;
    else
        w = o.F.(objname).w(end);
    end
end
w = w(:);

if isscalar(w)
    w = w * ones(size(J));
end

assert(numel(w) == numel(J),...
    'Number of weights (%d) must coincide the number of objective values (%d)',...
    numel(w),numel(J))

o.F.(objname).J = [ o.F.(objname).J ; J ];
o.F.(objname).w = [ o.F.(objname).w ; w ];

end