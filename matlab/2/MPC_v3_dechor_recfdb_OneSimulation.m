function R = MPC_v3_dechor_recfdb_OneSimulation(T,Tp,N,Iref,DirName,Name,args)
arguments
    T,Tp,N,Iref,DirName,Name
    args.FreeSpreadFromDate = datetime(2030,01,01)
    args.GenerateVideoFrames = false
    args.GenerateSparseFrames = false
    args.PanSimArgs = []
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 29. (2023a)
% 
% Decreasing, shrinking, narrowing, shortening horizon. 
% Feedback using the reconstructed, estimated epidemic state.

if exist('pansim','var')
    clear pansim
end
clear mex

%% TODO

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2024-02-26_Agens_Wild.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
P = Epid_Par.Get(Q);

%%

[~,Pmx_lezar] = max(T.IQ);
[~,Pmx_free] = min(T.IQ);

beta_free = T.TrRate(Pmx_free);
beta_lezar = T.TrRate(Pmx_lezar);

%% First policy

k0_Pmx = max(T.Pmx);
k0_PM = T(k0_Pmx,Vn.policy).Variables;
k0_TrRateExp = T.TrRate(k0_Pmx);

%%

Nr_Periods = N / Tp;

Start_Date = C.Start_Date;
End_Date = Start_Date + N;
P = P(isbetween(P.Date,Start_Date,End_Date),:);

t_sim = 0:N;
d_sim = Start_Date + t_sim;

FreeSpreadFromDay = days(args.FreeSpreadFromDate - Start_Date);
FreeSpreadFromPeriod = round(FreeSpreadFromDay / Tp) + 1;

%% Design controller

import casadi.*

[f,~,~,J] = epid.ode_model_8comp(C.Np);

%%

% Measured state
% xx = nan(J.nx,N+1);

% Policy measures
% PM = repmat(k0_PM,[N+1,1]);

% Measured beta
% beta_msd = nan(1,N+1);

%%

if isempty(args.PanSimArgs)
    % Load PanSim arguments
    PanSim_args = ps.load_PanSim_args;
else
    % PanSim_args allowed to be passed as an argument
    % 2024.05.27. (május 27, hétfő), 09:48
    PanSim_args = args.PanSimArgs;
end

%%%
% Create simulator object
DIR = fp.dir;
pansim = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
pansim.initSimulation(PanSim_args);

simout = pansim.runForDay(string(k0_PM));

%% Initialize timetable `R` (Results)

% Create a first row for `R`
R = [ ...
    hp.simout2table(simout)             ... PanSim output
    table(k0_Pmx,'VariableNames',"Pmx") ... Ordinal no. of applied policy measure (Pmx)
    hp.policy2table(k0_PM)              ... Applied policy measures in flags and values
    table(k0_TrRateExp,k0_TrRateExp,NaN,NaN,'VariableNames', ... Further variables:
        {'TrRateCmd','TrRateExp','TrRateRec','Ipred'})       ... Ipred: legacy
... array2table(nan(1,Nr_Periods),'VariableNames',Vn.IQk(0:Nr_Periods-1)) ... Planned IQ in the different phases
... array2table(nan(1,Nr_Periods),'VariableNames',Vn.TrRatek(0:Nr_Periods-1)) ... Estimated TrRate in the different phases
    array2table(nan(1,Nr_Periods),'VariableNames',Vn.Ipredk(0:Nr_Periods-1)) ... Predicted I in the different phases
... array2table(nan(1,Nr_Periods),'VariableNames',Vn.Hpredk(0:Nr_Periods-1)) ... Predicted H in the different phases
    array2table(nan(size(Vn.SLPIAHDR)),"VariableNames",Vn.SLPIAHDR+"r") ... Reconstructed state
    ];

% We assume that the initial state of the epidemic is known, therefore, the first
% reconstructed state is the actual state
R(:,Vn.SLPIAHDR + "r") = R(:,Vn.SLPIAHDR);

% Construct the full table by repeating the first row 
R = repmat(R,[N+1,1]);

% Append parameters to `R` as new colums
R = [R hp.param2table(P.Param)];

% Update the time flags of the timetable
R.Properties.RowTimes = d_sim;

% Remove values, which are not known yet
R.TrRate(2:end) = NaN;
R.I(2:end) = NaN;
R.Ir(2:end) = NaN;
% .... there would be more, but those are not relevant

% Update: 2024-03-05
w = Epid_Par.Interp_Sigmoid_v2(1, 0, 12,10, 1, N+1)';

% Append the reference trajectory to `R` as a new column
% Iref = Iref.*w + (1-w)*R.I(1);
R.Iref = Iref(1:height(R));

% Append `k` (control term) and `d` (day) to `R` as new columns
z = zeros(height(R),1);
R = addvars(R,z,z,'NewVariableNames',{'k','d'});

% Append the bounds for the transmission rate
beta_min = min(T.TrRate);
beta_max = max(T.TrRate);
R.TrRateBounds = repmat([beta_min beta_max],[height(R),1]);

% Append the estimated range of the transmission rate
beta_min_std = min(max(0,T.TrRate - 2*T.TrRateStd));
beta_max_std = max(T.TrRate + 2*T.TrRateStd);
R.TrRateRange = repmat([beta_min_std beta_max_std],[height(R),1]);

% Visualize_MPC_v8(R,0,0,"Tp",max(Tp,7));

%%

h = [2 2 3]*N/7;
wIerr = [ ones(1,h(1)) , Epid_Par.Sigmoid(1,0,h(2)) , zeros(1,h(3)) ] + 0.1;

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";

DIRf = fullfile(DIR,"Output","FullSim_" + string(Now));
if ~exist(DIRf,"dir")
    mkdir(DIRf)
end

for k = 0:Nr_Periods-1
    % This is the `k`th control term, which simulates `Tp` days

    N_MPC = Tp * (Nr_Periods-k);

    %%%
    % Current reference

    Idx = Tp*k+1:N;
    Iref_k = Iref(Idx+1)';

    %%%
    % Initial guess
    
    L_guess = Iref_k*0.15;
    P_guess = Iref_k*0.201;
    I_guess = Iref_k*0.401;
    A_guess = Iref_k*0.401;
    S_guess = C.Np - cumsum(L_guess);
    x_guess = [
        S_guess
        L_guess
        P_guess
        I_guess
        A_guess
        A_guess*0 % H
        A_guess*0 % D
        A_guess*0 % R
        ];
    x0 = R(Tp*k+1,Vn.SLPIAHDR + "r").Variables';
    x_fh = @(x_var) [x0 , x_var];
    
    %%%
    %  Create a matrix: M = [ 1 1 1 1 1 1 1 , 0 0 0 ...
    %                         0 0 0 0 0 0 0 , 1 1 1 ... ]
    idx = reshape(ones(Tp,1) * (1:Nr_Periods-k),N_MPC,1);
    I = eye(Nr_Periods-k);
    M = I(:,idx);
    
    beta_guess = ones(1,Nr_Periods-k) * beta_free;
    beta_fh = @(beta_var) beta_var * M;

    % -----------------------------------
    % Create optimization problem

    helper = Pcz_CasADi_Helper('SX');

    x_var = helper.new_var('x',size(x_guess),1,'str','full','lb',0);
    x = x_fh(x_var);

    ldx_ctrl = (k+1:Nr_Periods) < FreeSpreadFromPeriod;
    ldx_free = ~ldx_ctrl;
    
    beta_var = helper.new_var('beta',size(beta_guess),1,'str','full','lb',beta_min,'ub',beta_max);
    beta_mod = SX(beta_guess);
    beta_mod(find(ldx_ctrl)) = beta_var(find(ldx_ctrl));
    beta = beta_fh(beta_mod);
    
    % Enforce the state equations
    for i = 1:N_MPC
        x_kp1 = f.Fn(x(:,i),P.Param(k+i,:)',beta(i),0,0);
        helper.add_eq_con( x_kp1 - x(:,i+1) );
    end
    
    wI = wIerr(1:N_MPC);
    wI(( 1:numel(wI) ) + k*Tp > FreeSpreadFromDay) = 0;

    % Minimize the tracking error
    helper.add_obj('I_error',(x_var(J.I,:) - Iref_k).^2,wI);
    
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
    beta_sol(ldx_free) = beta_guess(ldx_free);
    x_sol = helper.get_value('x');
    x = x_fh(x_sol);

    % Update planned policy measures
    idx_sol = 1;
    for j = k:Nr_Periods-1

        [~,Pmx] = min(abs(T.TrRate - beta_sol(idx_sol)));

        Rp = T(Pmx,Vn.policy);
        for d = 1:Tp
            Idx = Tp*j+d;
            R.Pmx(Idx) = Pmx;
            R(Idx,Rp.Properties.VariableNames) = Rp;
            R.TrRateCmd(Idx) = beta_sol(idx_sol);
            R.TrRateExp(Idx) = T.TrRate(Pmx);
        end

        idx_sol = idx_sol + 1;
    end
    R = Vn.quantify_policy(R);

    % Update prediction
    Idx = Tp*k+1:height(R);
    R.(Vn.Ipredk(k))(Idx) = x(J.I,:)';
    % R.(Vn.Hpredk(k))(Idx) = x(J.H,:)';
    % R.(Vn.TrRatek(k))(Idx) = R.TrRateCmd(Idx);
    % R.(Vn.IQk(k))(Idx) = Vn.IQ(R(Idx,Vn.policy + "_Val"));

    fig = Visualize_MPC_v8(R,Tp*k+1,k,"Tp",max(Tp,7));
    drawnow
    if args.GenerateVideoFrames
        exportgraphics(fig,DIRf + "/" + sprintf('Per%02d_Day%03d_Rec',k,Tp*k+d) + ".png")
    end
    if args.GenerateSparseFrames
        exportgraphics(fig,DIRf + "/" + sprintf('Per%02d_Day%03d_Rec',k,Tp*k+d) + ".pdf",'ContentType',"vector")
    end

    % -----------------------------------
    % Simulate and collect measurement

    for d = 1:Tp
        simout = pansim.runForDay(string(R(Tp*k+d,Vn.policy).Variables));

        Idx = Tp*k+d;

        O = hp.simout2table(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
        R.k(Idx) = k;
        R.d(Idx) = d;

        if args.GenerateVideoFrames
            fig = Visualize_MPC_v8(R,Idx+1,k,"Tp",max(Tp,7));
            drawnow
            exportgraphics(fig,DIRf + "/" + sprintf('Per%02d_Day%03d_Adv',k,Tp*k+d) + ".png")
        end
    end

    Pend = (k+1)*Tp;
    if Pend >= 7
        R = rec_SLPIAHDR(R,Start_Date + [0,Pend],'PWConstBeta',true,'PWConstBetaTp',Tp);
    else
        R(:,Vn.SLPIAHDR + "r") = R(:,Vn.SLPIAHDR);
    end

end
clear pansim mex

R = rec_SLPIAHDR(R,'PWConstBeta',true,'PWConstBetaTp',Tp);
fig = Visualize_MPC_v8(R,N+1,Nr_Periods,"Tp",max(Tp,7));
drawnow
if args.GenerateVideoFrames
    exportgraphics(fig,DIRf + "/" + sprintf('Per%02d_Day%03d_Rec',Nr_Periods,N) + ".png")
end
if args.GenerateSparseFrames
    exportgraphics(fig,DIRf + "/" + sprintf('Per%02d_Day%03d_Rec',Nr_Periods,N) + ".pdf",'ContentType',"vector")
end

% writetimetable(R,DIR + "/A.xls","Sheet","Result");

fp = pcz_mfilename(mfilename("fullpath"));
Today = string(datetime('today','Format','uuuu-MM-dd'));
dirname = fullfile(fp.dir,"Output","Ctrl_" + Today,Name);
dirname = fullfile("/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output","Ctrl_" + Today,Name);
dirname = fullfile("/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output",DirName,Name);
if ~exist(dirname,'dir')
    mkdir(dirname)
end
Now = string(Now);

exportgraphics(fig,fullfile(dirname,Now + ".pdf"),'ContentType','vector');
exportgraphics(fig,fullfile(dirname,Now + ".jpg"),'ContentType','vector');

% R = rec_SLPIAHDR(R,'WeightBetaSlope',1e4,'PWConstBeta',false);
writetimetable(R,fullfile(dirname,Now + ".xls"),"Sheet","Result");

% movefile(DIR,DIR+"_Finalized")

end
