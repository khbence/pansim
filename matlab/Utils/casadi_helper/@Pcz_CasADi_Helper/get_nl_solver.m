function [NL_solver] = get_nl_solver(o,s)
arguments
    o
    s.Verbose = false;
end

%%
%  File: get_nl_solver.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 19. (2021a)
%

% Finalize helper object
o.construct;

NL_solver = Pcz_NL_Solver(o,'Verbose',s.Verbose);

end