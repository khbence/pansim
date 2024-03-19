function [R_,NewVariableNames] = rec_SLPIAHDR(R,DSpan,Np,args)
arguments
%%
    R
    DSpan = R.Date([1,end]);

    % Population in the simulator
    Np = C.Np;

    args.WeightIError = 1;
    args.WeightBetaSlope = 1e6;
    args.Visualize = false;
    args.PWConstBeta = true;
    args.BetaRange = [0.01,0.5];
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 12. (2023a)

%%

Start_Date = DSpan(1);
End_Date = DSpan(2);

ldx = isbetween(R.Date,Start_Date,End_Date);
R_ = R;

R = R(ldx,:);

N = height(R) - 1;

%% Construct optimization

import casadi.*
helper = Pcz_CasADi_Helper('SX');

K = Epid_Par.GetK;
[f,~,~,J] = epid.ode_model_8comp(Np);

% -----------------------------------
% Reference values

x_PanSim = R(:,Vn.SLPIAHDR).Variables';

% -----------------------------------
% Create optimization problem

x0 = x_PanSim(:,1);
x_guess = x_PanSim(:,2:end) + randn(size(x_PanSim)-[0 1]);
x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0,'ub',Np);
x_fh = @(x_var) [x0 , x_var];
x = x_fh(x_var);

beta_min = args.BetaRange(1);
beta_max = args.BetaRange(2);
if args.PWConstBeta 
    [IQ_vals,idx,Idx] = unique(R.IQ(1:end-1));
    
    beta_guess = groupsummary(R.TrRate(1:end-1),Idx,"mean")';
    beta_guess = min(max(beta_min,beta_guess),beta_max);
    beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);

    Idx(end+1) = Idx(end);
    beta_fh = @(beta_var) beta_var(Idx');
    beta = beta_fh(beta_var);
else
    beta0 = R.TrRate(1);
    beta_guess = R.TrRate(2:end-1)';
    beta_guess = beta_guess + randn(size(beta_guess))*0.05;
    beta_guess = min(max(beta_min,beta_guess),beta_max);
    beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);
    
    beta_fh = @(beta_var) [ beta0 , beta_var , beta_var(end) ];
    beta = beta_fh(beta_var);

    helper.add_obj('beta_slope',sumsqr(diff(beta)),args.WeightBetaSlope);
end

% Enforce the state equations
Param = R(:,Vn.params).Variables;
for i = 1:N
    x_kp1 = f.Fn(x(:,i),Param(i,:)',beta(i),0,0);
    helper.add_eq_con( x_kp1 - x(:,i+1) );
end

% Minimize the tracking error
% helper.add_obj('S_error',sumsqr(x(J.S,:) - R.S'),0.01);
% helper.add_obj('L_error',sumsqr(x(J.L,:) - R.L'),1);
% helper.add_obj('P_error',sumsqr(x(J.P,:) - R.P'),1);
helper.add_obj('I_error',sumsqr(x(J.I,:) - R.I'),args.WeightIError);
% helper.add_obj('A_error',sumsqr(x(J.A,:) - R.A'),1);
% helper.add_obj('H_error',sumsqr(x(J.H,:) - R.H'),100);
% helper.add_obj('D_error',sumsqr(x(J.D,:) - R.D'),10);
% helper.add_obj('R_error',sumsqr(x(J.R,:) - R.R'),0.01);

% Construct the nonlinear solver object
NL_solver = helper.get_nl_solver("Verbose",true);

% Retain the mapping for the free variables, which allows to construct an
% initial guess vector for the nonlinear solver.
Fn_var = NL_solver.helper.gen_var_mfun;
sol_guess = full(Fn_var(x_guess,beta_guess));

% Solve the control optimization problem
ret = NL_solver.solve([],sol_guess);

% Get the solution
beta_sol = helper.get_value('beta');
x_sol = helper.get_value('x');

R.TrRateRec = beta_fh( beta_sol )';
x = x_fh( x_sol );

R(:,Vn.SLPIAHDR + "r") = array2table(x');


% 2024.02.29. (február 29, csütörtök), 14:07
NewVariableNames = [ "TrRateRec" , Vn.SLPIAHDR+"r" ];
for vn = NewVariableNames
    if ~ismember(vn,R_.Properties.VariableNames)
        R_ = addvars(R_,nan(height(R_),1),'NewVariableNames',vn);
    end
end

R_(ldx,:) = R(:,R_.Properties.VariableNames);

%%

if args.Visualize
    Fig = figure(61512);
    delete(Fig.Children)
    
    nexttile
    plot(R.Date,[R.TrRate , R.TrRateRec])
    title('Transmission rate')
    
    for i = 1:J.nx
        nexttile
        plot(R.Date,[x_PanSim(i,:)' , x(i,:)'])
        title(Vn.SLPIAHDR(i))
    end
end


end
