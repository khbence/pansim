function construct(o)
%%
%  File: construct.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 19. (2020b)
%

import casadi.*

% Cummulate objective functions
o.f = 0;
fns = fieldnames(o.F);
for i = 1:numel(fns)
    Ji = sum(o.F.(fns{i}).J .* o.F.(fns{i}).w);
    o.f = o.f + Ji;
    o.F.(fns{i}).f_J = Function('f_J',{o.x,o.p},{Ji});
    o.F.(fns{i}).f_Jk = Function('f_Jk',{o.x,o.p},{o.F.(fns{i}).J});
end

% Cummulate objective functions
o.g = [];
o.lbg = [];
o.ubg = [];
fns = fieldnames(o.G);
for i = 1:numel(fns)
    o.g = [
        o.g
        o.G.(fns{i}).g(:)
        ];
    
    o.lbg = [
        o.lbg
        o.G.(fns{i}).lb(:)
        ];
    
    o.ubg = [
        o.ubg
        o.G.(fns{i}).ub(:)
        ];
end

end
