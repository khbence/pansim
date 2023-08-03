%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%#ok<*CLALL>

Np = 179500;

%% Load policy measures

matrix = readMatrixFromFile('matlab/Data/data2.txt');
T = cell2table(matrix,'VariableNames',["TP","PL","CF","SO","QU","MA","Week","Beta"]);
T.Week = categorical(T.Week);

ldx = T.Week == "WEEK0";
Idx = [ find(ldx) ; numel(T.Week)+1 ];
Idx = num2cell([Idx(1:end-1) , Idx(2:end)],2);
Idx = cellfun(@(Idx) {Idx(1):Idx(2)-1},Idx);
Mean_Beta = cellfun(@(Idx) mean(T.Beta(Idx)),Idx);

T = removevars(T(ldx,:),"Week");
T.Beta = Mean_Beta;

%% Initialize simulator

N_Sim = 100;
PanSim_args = [
    "panSim", "-r", " ", ...
     "--quarantinePolicy", "0", ...
     "-k", "0.00041", ...
     "--progression", "inputConfigFiles/progressions_Jun17_tune/transition_config.json", ...
     "--closures", "inputConfigFiles/emptyRules.json", ...
     "-A", "inputConfigFiles/agentTypes_3.json", ...
     "-a", "inputRealExample/agents1.json", ...
     "-l", "inputRealExample/locations0.json", ...
     "--infectiousnessMultiplier", "0.98,1.81,2.11,2.58,4.32,6.8,6.8", ...
     "--diseaseProgressionScaling", "0.94,1.03,0.813,0.72,0.57,0.463,0.45"
     ];
save PanSim_args PanSim_args T N_Sim Np
clear all

%% Simulation: no restrictions

load PanSim_args.mat

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

beta0 = max(T.Beta);
[~,Idx_Closest] = min(abs(T.Beta - beta0));
Policy_measures = string(T(Idx_Closest,1:end-1).Variables);

nx = 4;
x = zeros(nx,N_Sim+1);
beta = [ zeros(1,N_Sim) beta0 ];

for k = 2:N_Sim+1

    % Execute PanSim
    simout = obj.runForDay(Policy_measures);
    [simx,simbeta] = get_SEIRb(simout,Np);    

    x(:,k) = simx;
    beta(k-1) = simbeta;

end

save PanSim_min x beta beta0 Policy_measures

clear all

%% Simulation: strict restrictions

load PanSim_args.mat

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

beta0 = min(T.Beta);
[~,Idx_Closest] = min(abs(T.Beta - beta0));
Policy_measures = string(T(Idx_Closest,1:end-1).Variables);

nx = 4;
x = zeros(nx,N_Sim+1);
beta = [ zeros(1,N_Sim) beta0 ];

for k = 2:N_Sim+1

    % Execute PanSim
    simout = obj.runForDay(Policy_measures);
    [simx,simbeta] = get_SEIRb(simout,Np);    

    x(:,k) = simx;
    beta(k-1) = simbeta;

end

save PanSim_max x beta beta0 Policy_measures

clear all

%% Visualize

load PanSim_min.mat
x_min = x;
beta_min = beta;
pm_min = strjoin(strtrim(Policy_measures),', ') + " -- no restrictions";

load PanSim_max.mat
x_max = x;
beta_max = beta;
pm_max = strjoin(strtrim(Policy_measures),', ') + " -- strictest restrictions";

load PanSim_args.mat
t = 0:N_Sim;

fig = figure(12);
fig.Position(3:4) = [716 905];
Tl = tiledlayout(2,1,'Padding','compact','TileSpacing','tight');

ax = nexttile;
plot(t,[x_min(3,:);x_max(3,:)]'/Np*100)
Leg = legend(pm_min,pm_max);
title 'All infected (percentage)'
xlabel days
ylabel 'Inf/Population * 100'
grid on

ylim([0 9])

ax = nexttile;
plot(t,[beta_min;beta_max]')
Leg = legend(pm_min,pm_max);
title 'Estimated beta'
xlabel days
ylabel 'beta'
grid on
