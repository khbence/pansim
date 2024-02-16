%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)
% 
% Decreasing, shrinking, narrowing, shortening horizon

%%

clear all

matname = 'matlab/2/PM_Indices.mat';
s = load(matname);
s.Actual_Pmx = s.Actual_Pmx + 1;

save(matname,"-struct","s")

PM_Indices = s.PM_Indices(s.Actual_Pmx,:);

%% Load parameters

% fp = pcz_mfilename(mfilename('fullpath'));
% DIR_Data = fullfile(fp.pdir,'Data');
% 
% filename = fullfile(DIR_Data,'data2.txt');
% savename = fullfile(DIR_Data,'Policy_measures.mat');
% T = hp.load_policy_measures('filename',filename,'savename',savename,'reset',true);
% T(isnan(T.Pmx),:) = [];

T = hp.load_policy_measures;

nP = max(T.Pmx);

Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2023-12-19_JN1.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q(["Transient","Original","Future"],:);
P = Epid_Par.Get(Q);

%%

N = s.N;
Nr_Periods = s.Nr_Periods;
Tp = s.Tp;

Start_Date = C.Start_Date;
End_Date = Start_Date + N;
P = P(isbetween(P.Date,Start_Date,End_Date),:);

t_sim = 0:N;
d_sim = Start_Date + t_sim;

%%

R = [ ...
    hp.simout2table(nan(size(Vn.simout))) ...
    hp.quantify_policy(T(1,Vn.policy)) ...
    table(NaN,NaN,NaN,NaN,'VariableNames',...
        {'TrRateCmd','TrRateExp','TrRateRec','Ipred'}) ...
    array2table(nan(size(Vn.SLPIAHDR)),"VariableNames",Vn.SLPIAHDR+"r")
    ];
R = repmat(R,[N+1,1]);
R = [R hp.param2table(P.Param)];
R.Properties.RowTimes = d_sim;
R.TrRate(2:end) = NaN;
R.I(2:end) = NaN;
R.Iref = nan(height(R),1);
R.TrRateBounds = zeros(height(R),1) + [0.02 0.5];

z = zeros(height(R),1);
R = addvars(R,z,z,'NewVariableNames',{'k','d'});

%%

% Load PanSim arguments
PanSim_args = ps.load_PanSim_args;

%%%
% Create simulator object
DIR = fileparts(mfilename('fullpath'));
obj = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

% First step
PM0 = T(PM_Indices(1),Vn.policy);
simout = obj.runForDay(string(PM0.Variables));
O = hp.simout2table(simout);
R(1,O.Properties.VariableNames) = O;

%%

for k = 0:Nr_Periods-1
    PM = T(PM_Indices(k+1),Vn.policy);
    for d = 1:Tp
        R(Tp*k+d,PM.Properties.VariableNames) = PM;
    end
end
R = hp.quantify_policy(R);

for k = 0:Nr_Periods-1

    PM = R(k*Tp+1,Vn.policy);
        
    % -----------------------------------
    % Simulate and collect measurement

    for d = 1:Tp
        simout = obj.runForDay(string(PM.Variables));

        Idx = Tp*k+d;

        O = hp.simout2table(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
        R.k(Idx) = k;
        R.d(Idx) = d;

        Visualize_MPC(R,Idx+1);
    
        drawnow

        % exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,Tp*k+d) + ".png")
    end
    R = rec_SLPIAHDR(R,Start_Date + [0,(k+1)*Tp],C.Np);

end

fig = Visualize_MPC(R,N+1);
drawnow

filename = s.DIR + "/A" + num2str(s.Actual_Pmx) + ".xls";
writetimetable(R,filename,"Sheet","Results");
exportgraphics(fig,s.DIR + "/Fig" + num2str(s.Actual_Pmx) + ".jpg");

