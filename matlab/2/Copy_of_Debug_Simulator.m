%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 29. (2023a)
% 
% Decreasing, shrinking, narrowing, shortening horizon. 
% Feedback using the reconstructed, estimated epidemic state.


Name = "Debug";

N = 6*7*4;
Tp = 7;

fp = pcz_mfilename(mfilename("fullpath"));
fname = fullfile(fp.dir,"Output","Summary_2024-02-27__2.xls");
opts = detectImportOptions(fname);
opts = setvartype(opts,Vn.policy,"categorical");
T = readtable(fname,opts);

Iq_k = T.Properties.VariableNames(startsWith(T.Properties.VariableNames,'Iq_'));
T.Iq = T(:,Iq_k).Variables;
T(:,Iq_k) = [];

T.Pmx = (1:height(T))';

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

% [~,Pmx_Mindent_lezar] = max(vecnorm(T.Iq,1,2));
% [~,Pmx_Mindent_felenged] = min(vecnorm(T.Iq,1,2));

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

%%

% Measured state
% xx = nan(J.nx,N+1);

% Policy measures
% PM = repmat(k0_PM,[N+1,1]);

% Measured beta
% beta_msd = nan(1,N+1);

%%
PmxA = ones(1,2);
PmxA(1) = T.Pmx(find(T.IQ == 100000));
PmxA(2) = T.Pmx(find(T.IQ == 100000));
PmxA(3) = T.Pmx(find(T.IQ == 100002));
PmxA(4) = T.Pmx(find(T.IQ == 100040));
PmxA(5) = T.Pmx(find(T.IQ == 100200));
PmxA(6) = T.Pmx(find(T.IQ == 101000));
PmxA(7) = T.Pmx(find(T.IQ == 110000));
PmxA(8) = T.Pmx(find(T.IQ == 400000));
PmxA(9) = T.Pmx(find(T.IQ == 411240));
SName = ["" , "FreeSpread" , flip(Vn.policy) , "AllButMA"];

PmxA(1) = T.Pmx(find(T.IQ == 100002));
PmxA(2) = T.Pmx(find(T.IQ == 100002));
PmxA(3) = T.Pmx(find(T.IQ == 100000));
PmxA(4) = T.Pmx(find(T.IQ == 100042));
PmxA(5) = T.Pmx(find(T.IQ == 100202));
PmxA(6) = T.Pmx(find(T.IQ == 101002));
PmxA(7) = T.Pmx(find(T.IQ == 110002));
PmxA(8) = T.Pmx(find(T.IQ == 400002));
PmxA(9) = T.Pmx(find(T.IQ == 411242));
SName = ["" , "FreeSpread" , flip(Vn.policy) , "AllButMA"];

for a = 2:9

    Pmx = PmxA([1,a]);

    % Load PanSim arguments
    PanSim_args = ps.load_PanSim_args;
    
    %%%
    % Create simulator object
    DIR = fileparts(mfilename('fullpath'));
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
        array2table(nan(1,Nr_Periods),'VariableNames',Vn.IQk(0:Nr_Periods-1)) ... Planned IQ in the different phases
        array2table(nan(1,Nr_Periods),'VariableNames',Vn.TrRatek(0:Nr_Periods-1)) ... Estimated TrRate in the different phases
        array2table(nan(1,Nr_Periods),'VariableNames',Vn.Ipredk(0:Nr_Periods-1)) ... Predicted I in the different phases
        array2table(nan(1,Nr_Periods),'VariableNames',Vn.Hpredk(0:Nr_Periods-1)) ... Predicted H in the different phases
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
    R.Iref = R.I*0 + C.Np*0.02;
    
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
    
    Visualize_MPC_v3(R,0,0,"Tp",max(Tp,7));
    
    %%
    
    h = [2 2 3]*N/7;
    wIerr = [ ones(1,h(1)) , Epid_Par.Sigmoid(1,0,h(2)) , zeros(1,h(3)) ] + 0.1;
    
    Now = datetime;
    Now.Format = "uuuu-MM-dd_HH-mm";


    for k = 0:Nr_Periods-1
    
        Pmx_k = Pmx(mod(k,2)+1);
        PM = T(Pmx_k,Vn.policy);
        TrRateCmd = T.TrRate(Pmx_k);
            
        for d = 1:Tp
            Idx = Tp*k+d;
            R.k(Idx) = k;
            R.d(Idx) = d;
            R.Pmx(Idx) = Pmx_k;
            R.TrRateCmd(Idx) = TrRateCmd;
            R(Idx,PM.Properties.VariableNames) = PM;            
        end
    end
    R = Vn.quantify_policy(R);
    
    for k = 0:Nr_Periods-1
    
        Pmx_k = Pmx(mod(k,2)+1);
        PM = T(Pmx_k,Vn.policy);
            
        for d = 1:Tp
            simout = pansim.runForDay(string(PM.Variables));
    
            Idx = Tp*k+d;        
            O = hp.simout2table(simout);
            R(Idx+1,O.Properties.VariableNames) = O;
        
            if mod(Idx,21) == 0 && Idx > 7
                R = rec_SLPIAHDR(R,Start_Date + [0,Idx],C.Np,'WeightBetaSlope',1e4);
                fig = Visualize_MPC_v3(R,Idx+1,k,"Tp",max(Tp,7));
                drawnow
            end
        end
    end
    clear pansim mex
    
    fig = Visualize_MPC_v3(R,N+1,Nr_Periods,"Tp",max(Tp,7));
    
    fp = pcz_mfilename(mfilename("fullpath"));
    dirname = fullfile(fp.dir,"Output","Ctrl_2024-02-27",Name);
    if ~exist(dirname,'dir')
        mkdir(dirname)
    end
    fname = SName(a) + "_Tp" + sprintf('%02d',Tp) + "_" + string(Now);
    
    exportgraphics(fig,fullfile(dirname,fname + ".pdf"),'ContentType','vector');
    exportgraphics(fig,fullfile(dirname,fname + ".jpg"),'ContentType','vector');
    
    R = rec_SLPIAHDR(R,'WeightBetaSlope',1e4);
    R_red = R(:,["Pmx",Vn.policy,"IQ","Iq",Vn.SLPIAHDR,Vn.simout,"TrRate",Vn.SLPIAHDR+"r","TrRateRec"]);
    writetimetable(R_red,fullfile(dirname,fname + ".xls"),"Sheet","Result");

end
