function Step1_Generate_Simulation_Data
arguments
end

fp = pcz_mfilename(mfilename("fullpath"));
DIR_Input = fullfile(fp.dir,'Input');
DIR_Output = fullfile(fp.dir,'Output');
MAT_PM_Allcomb = fullfile(DIR_Input,'PM_Allcomb.mat');
MAT_PM_combi = @(i) fullfile(DIR_Input,sprintf('PM_comb%04d.mat',i));

Today = string(datetime('today','Format','uuuu-MM-dd'));
DIR = fullfile(DIR_Output,"Allcomb_" + Today);

if ~exist(DIR_Input,"dir")
    mkdir(DIR_Input)
end

if ~exist(DIR_Output,"dir")
    mkdir(DIR_Output)
end

if ~exist(DIR,"dir")
    mkdir(DIR);
end

%% PM_Allcomb.mat

T = Vn.allcomb;
nP = height(T);

Tp = 28; 
Nr_Periods = 6;
save(MAT_PM_Allcomb,"DIR","T","Tp","Nr_Periods")

%% PM_comb000i.mat

N_sim = 5; % 1000;
Pmx_Perms = zeros(N_sim,Nr_Periods);
for i = 1:N_sim
    Perm = randperm(nP,Nr_Periods);
    Pmx_Perms(i,:) = Perm;
    save(MAT_PM_combi(i),"Perm")
end

fig = figure(12);
histogram(Pmx_Perms(:)-0.5,'BinEdges',(0:nP));

for i = 1:N_sim
    Simulate_Interventions(i);
end
end

function Simulate_Interventions(PM_comb_Idx)
%%

if exist('pansim','var')
    clear pansim
end
clear mex

%% Load input data

fp = pcz_mfilename(mfilename("fullpath"));
DIR_Input = fullfile(fp.dir,'Input');
DIR_Output = fullfile(fp.dir,'Output');
MAT_PM_Allcomb = fullfile(DIR_Input,'PM_Allcomb.mat');
MAT_PM_combi = @(i) fullfile(DIR_Input,sprintf('PM_comb%04d.mat',i));

s = load(MAT_PM_Allcomb);
T = s.T;
Tp = s.Tp;
Nr_Periods = s.Nr_Periods;
N = Nr_Periods*Tp;

DIR = s.DIR;
MAT_Results = fullfile(DIR,sprintf("A%04d.xls",PM_comb_Idx));

if exist(MAT_Results,"file")
    fprintf('File `%s` already exists! Exiting ...\n',MAT_Results)
    return
end

s = load(MAT_PM_combi(PM_comb_Idx));
PM_Indices = s.Perm;

clear s

%%

Start_Date = C.Start_Date;

t_sim = 0:N;
d_sim = Start_Date + t_sim;

%%

R = [ ...
    hp.simout2table(nan(size(Vn.simout))) ...
    table(PM_Indices(1),'VariableNames',"Pmx") ...
    Vn.quantify_policy(T(PM_Indices(1),Vn.policy)) ...
    ];
R = repmat(R,[N+1,1]);
R.Properties.RowTimes = d_sim;

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

tic
for k = 0:Nr_Periods-1

    Pmx = PM_Indices(k+1);
    PM = T(Pmx,Vn.policy);
        
    for d = 1:Tp
        simout = pansim.runForDay(string(PM.Variables));

        Idx = Tp*k+d;
        R.k(Idx) = k;
        R.d(Idx) = d;
        R.Pmx(Idx) = Pmx;
        R(Idx,PM.Properties.VariableNames) = PM;
        
        O = hp.simout2table(simout);
        R(Idx+1,O.Properties.VariableNames) = O;
    end
end
toc
clear pansim mex

R = Vn.quantify_policy(R);

writetimetable(R,MAT_Results,"Sheet","Results");
end