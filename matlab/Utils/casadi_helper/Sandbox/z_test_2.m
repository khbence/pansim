%%
%  File: z_test_2.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 19. (2020b)
%

G_reset

%%

Timer_vl66 = pcz_dispFunctionName('Construct optimization problem');

    import casadi.*

    n = 3;
    N = 10;
    m = 6;

    A = [1 0 1 ; 1 1 0 ; 0 1 1 ];
    B = [1 0 0]';

    helper = Pcz_CasADi_Helper;

    x = helper.new_var('x',[n,N],m);
    u = helper.new_var('u',[1,N],m,lb=-5,ub=5);

    Q = helper.new_par('Q',n,str="sym");
    R = helper.new_par('R',1,str="sym");

    % u = randn(1,N);

    rng(1)
    x0 = cell(1,m);
    for i = 1:m
        x0{i} = randn(n,1);
        xi = [x0{i}, x{i}];
        for k = 1:N
            h = -xi(:,k+1) + A*xi(:,k) + B*u{i}(:,k);
            helper.add_eq_con(h);
        end
    end

    for i = 1:m
        Ji_x = sum(x{i} .* (Q*x{i}),1);
        Ji_u = sum(u{i} .* (R*u{i}),1);
        helper.add_obj('Q',Ji_x);
        helper.add_obj('R',Ji_u);
    end
    
    solver = helper.get_qp_solver;

    options = optimset('Display','iter');
    options = optimset();

pcz_dispFunctionEnd(Timer_vl66);

%%

Timer_EqVV = pcz_dispFunctionName('Call the solver object');

    solver.set_param(Q,eye(n));
    solver.set_param(R,eye(1));
    sol = solver.solve(options);

    exitflag = sol.exitflag;
    pcz_dispFunction_scalar(exitflag)
    
pcz_dispFunctionEnd(Timer_EqVV);

%%

Timer_EqVV = pcz_dispFunctionName('Call the solver object for different parameters');

    solver.set_param(Q,eye(n)*10);
    solver.set_param(R,eye(1)/10);
    sol = solver.solve(options);

    exitflag = sol.exitflag;
    pcz_dispFunction_scalar(exitflag)

pcz_dispFunctionEnd(Timer_EqVV);

%%

x_ = solver.get_value(x);
u_ = solver.get_value(u);

F = solver.get_obj;

F.Q.Jk

%%%
% Symbolic check

%{ 

    x_sym = cell(1,m);
    x_var = cell(1,m);
    for i = 1:m
        x_sym{i} = sym(sprintf('x%%d_%%d_%d',i),[N,n]).';
        x_var{i} = x_sym{i}(:);
    end

    u_sym = sym('u',[N,1]).';
    u_var = u_sym(:);

    vars = [ vertcat(x_var{:}) ; u_var ];

    J = 0.5 * vars.' * sol.H * vars + vars.' * sol.f;

    h = vpa(sol.B * vars - sol.c,2);

%}

%%%
% Numerical check

fig = figure(1);
delete(fig.Children);
axes('Parent',fig);
hold on

for i = 1:m
    xi = [x0{i}, full(x_{i})];
    ui = full(u_{i});
    for k = 1:N
        h = -xi(:,k+1) + A*xi(:,k) + B*ui(:,k);
    end
    
    Pl = plot(xi(1,:),xi(2,:),'.-');
    Pl = plot(xi(1,1),xi(2,1),'.','MarkerSize',20,'Color',Pl.Color);
end


