function R = Sim_interventions(T,PM_Indices,Tp,DirName,Name,args)
arguments
    T,PM_Indices,Tp,DirName,Name
    args.Visualize = true;
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
% 
% `Idx` should be set first

%%

if exist('pansim','var')
    clear pansim
end
clear mex

%%

Nr_Periods = numel(PM_Indices);
N = Nr_Periods * Tp;

Start_Date = C.Start_Date;
End_Date = Start_Date + N;

t_sim = 0:N;
d_sim = Start_Date + t_sim;

%% Load parameters

fp = pcz_mfilename(mfilename("fullpath"));
Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2024-02-26_Agens_Wild.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,Start_Date,End_Date),:);

%%

R = [ ...
    hp.simout2table(nan(size(Vn.simout))) ...
    Vn.quantify_policy(T(1,Vn.policy)) ...
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
pansim = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
pansim.initSimulation(PanSim_args);

%%

% First step
PM0 = T(PM_Indices(1),Vn.policy);
simout = pansim.runForDay(string(PM0.Variables));
O = hp.simout2table(simout);
R(1,O.Properties.VariableNames) = O;

%%

for k = 0:Nr_Periods-1
    PM = T(PM_Indices(k+1),Vn.policy);
    for d = 1:Tp
        R(Tp*k+d,PM.Properties.VariableNames) = PM;
    end
end
R = Vn.quantify_policy(R);

for k = 0:Nr_Periods-1

    PM = R(k*Tp+1,Vn.policy);
        
    % -----------------------------------
    % Simulate and collect measurement

    for d = 1:Tp
        simout = pansim.runForDay(string(PM.Variables));

        Idx = Tp*k+d;

        O = hp.simout2table(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
        R.k(Idx) = k;
        R.d(Idx) = d;
    end
    R = rec_SLPIAHDR(R,Start_Date + [0,(k+1)*Tp],C.Np);

    if args.Visualize
        fig = Visualize_MPC(R,Idx+1,"Tp",Tp);    
        drawnow
        % exportgraphics(fig,DIR + "/" + sprintf('Per%02d_Day%03d',k,Tp*k+d) + ".png")
    end
    
end
clear pansim mex

if args.Visualize
    fig = Visualize_MPC(R,N+1,"Tp",Tp);
    drawnow
    
    dirname = fullfile("/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output",DirName,Name);
    if ~exist(dirname,'dir')
        mkdir(dirname)
    end
    
    Now = datetime;
    Now.Format = "uuuu-MM-dd_HH-mm";
    Now = string(Now);
    
    writetimetable(R,fullfile(dirname,Now + ".xls"),"Sheet","Results");
    exportgraphics(fig,fullfile(dirname,Now + ".pdf"),'ContentType','vector');
    exportgraphics(fig,fullfile(dirname,Now + ".jpg"),'ContentType','vector');
end

end
