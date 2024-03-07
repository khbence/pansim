classdef Pcz_QP_Solver < Pcz_Abstract_Solver
%%
%  File: Pcz_QP_Solver.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com)
%
%  Created on 2021. April 19. (2020b)
%

properties

f_QP_matrices

end

methods

    function o = Pcz_QP_Solver(helper,f_QP_matrices)

        o = o@Pcz_Abstract_Solver(helper);
        o.f_QP_matrices = f_QP_matrices;
        
    end

    function sol = solve_verbose(o,x0,args)
        Timer_srfw = pcz_dispFunctionName('Pcz_QP_Solver::solve');

            o.check_params
        
            Timer_0hi4 = pcz_dispFunctionName('Parse arguments');

                if nargin < 2
                    x0 = [];
                end

                if nargin < 3
                    if isstruct(x0)
                        args = x0;
                        x0 = [];
                    else
                        args = mskoptimset('Diagnostics','on','Display','iter');
                    end
                end

            pcz_dispFunctionEnd(Timer_0hi4);

            % -----------------------------------------------------------------

            Timer_vl66 = pcz_dispFunctionName('Evaluate CasADi Function');

                [H,f,A,b,B,c,l,u] = o.f_QP_matrices(o.helper.p_val);

            pcz_dispFunctionEnd(Timer_vl66);

            % -----------------------------------------------------------------

            Timer_jdgr = pcz_dispFunctionName('Convert CasADi objects to numeric');

                H = sparse(H);
                f = sparse(f);
                A = sparse(A);
                b = sparse(b);
                B = sparse(B);
                c = sparse(c);
                l = sparse(l);
                u = sparse(u);

            pcz_dispFunctionEnd(Timer_jdgr);

            % -----------------------------------------------------------------

            Timer_CIhN = pcz_dispFunctionName('Solve QP problem (quadprog)');

                % quadprog(H,f,A,b,B,c,l,u,x0,args)
                [x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,B,c,l,u,x0,args);

            pcz_dispFunctionEnd(Timer_CIhN);

            % -----------------------------------------------------------------

            Timer_owid = pcz_dispFunctionName('Collect results');

                sol = v2struct(x,fval,exitflag,output,lambda);

                sol.H = H;
                sol.f = f;
                sol.A = A;
                sol.b = b;
                sol.B = B;
                sol.c = c;
                sol.l = l;
                sol.u = u;

                fns = fieldnames(o.helper.P);
                for i = 1:numel(fns)
                    o.helper.P.(fns{i}).set = false;
                end
                
                o.helper.x_sol = x;
                
            pcz_dispFunctionEnd(Timer_owid);

        pcz_dispFunctionEnd(Timer_srfw);
    end

    function sol = solve(o,x0,args)
    %%
        if nargin < 2
            x0 = [];
        end

        if nargin < 3
            if isstruct(x0)
                args = x0;
                x0 = [];
            else
                args = struct;
            end
        end

        [H,f,A,b,B,c,l,u] = o.f_QP_matrices(o.helper.p_val);
        H = sparse(H);
        f = sparse(f);
        A = sparse(A);
        b = sparse(b);
        B = sparse(B);
        c = sparse(c);
        l = sparse(l);
        u = sparse(u);

        % quadprog(H,f,A,b,B,c,l,u,x0,args)
        [x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,B,c,l,u,x0,args);

        sol = v2struct(x,fval,exitflag,output,lambda);
        sol.H = H;
        sol.f = f;
        sol.A = A;
        sol.b = b;
        sol.B = B;
        sol.c = c;
        sol.l = l;
        sol.u = u;

        fns = fieldnames(o.helper.P);
        for i = 1:numel(fns)
            o.helper.P.(fns{i}).set = false;
        end

        o.helper.x_sol = x;

    end

    function [x,exitflag] = solve_in_sim(o,p,x0,args)
    arguments
        o,p,x0
        args = {}
    end
    %%
        [H,f,A,b,B,c,l,u] = o.f_QP_matrices(p);
        H = sparse(H);
        f = sparse(f);
        A = sparse(A);
        b = sparse(b);
        B = sparse(B);
        c = sparse(c);
        l = sparse(l);
        u = sparse(u);

        assert(all(full(l <= u)),'Condition `l <= x <= u` infeasible');

        % quadprog(H,f,A,b,B,c,l,u,x0,args)
        [x,~,exitflag] = quadprog(H,f,A,b,B,c,l,u,x0,args);
        o.helper.x_sol = x;
        o.helper.p_val = p;

    end

end

end
