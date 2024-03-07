classdef Pcz_NL_Solver < Pcz_Abstract_Solver
%%
%  File: Pcz_NL_Solver.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com)
%
%  Created on 2021. April 19. (2020b)
%

properties

solution = []

nlp
solver

f_bounds

end

methods

    function o = Pcz_NL_Solver(helper,s)
    arguments
        helper
        s.Verbose = false;
    end


        import casadi.*
        
        o = o@Pcz_Abstract_Solver(helper);        
        
        o.nlp = struct(...
            x = helper.x,...
            p = helper.p,...
            f = helper.f,...
            g = [
                helper.g
                helper.h
                ]...
            );
        
        opts = struct;
        % opts.ipopt.max_iter = 50000;
        opts.verbose = false;
        opts.verbose_init = false;
        opts.print_time = false;
        opts.print_in = false;
        opts.print_out = false;
        if ~s.Verbose
            opts.ipopt.print_level = 0; % Comment this 
        end

        o.solver = nlpsol('solver','ipopt',o.nlp,opts);

        bh = zeros(size(helper.h));
        
        lbg = [
            helper.lbg
            bh
            ];
        
        ubg = [
            helper.ubg
            bh
            ];
        
        o.f_bounds = Function('f_bounds',...
            {helper.p},...
            {helper.lbx,helper.ubx,lbg,ubg},...
            {'p'},...
            {'lbx','ubx','lbg','ubg'});

    end

    function sol = solve(o,p_val,x0)
        arguments
            o
            p_val = '[empty]';
            x0 = [];
        end
        
        % Timer_srfw = pcz_dispFunctionName('Pcz_NL_Solver::solve');

            if ischar(p_val)
                o.check_params
                p_val = o.helper.p_val;
            else
                o.helper.p_val = p_val;
            end

            [lbx,ubx,lbg,ubg] = o.f_bounds(p_val);                

            % -----------------------------------------------------------------

            % Timer_CIhN = pcz_dispFunctionName('Solve nonlinear problem (ipopt)');

                if isempty(x0)
                    x0 = zeros(size(o.helper.x));
                end
            
                sol = o.solver('x0',x0,'p',p_val,'lbx',lbx,'ubx',ubx,'lbg',lbg,'ubg',ubg);

            % pcz_dispFunctionEnd(Timer_CIhN);

            % -----------------------------------------------------------------

            % Timer_owid = pcz_dispFunctionName('Collect results');
            
                if ischar(p_val)
                    fns = fieldnames(o.helper.P);
                    for i = 1:numel(fns)
                        o.helper.P.(fns{i}).set = false;
                    end
                end
                
                o.solution = sol;
                o.helper.x_sol = sol.x;
                
            % pcz_dispFunctionEnd(Timer_owid);

        % pcz_dispFunctionEnd(Timer_srfw);
    end

end

end
