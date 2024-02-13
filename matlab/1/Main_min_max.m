%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. August 3. (2023a)
%
%    THIS SCRIPT SHOULD BE CALLED IN THE ROOT DIRECTORY
% 
%#ok<*CLALL>

% Population in the simulator
Np = 179500;

T = load_policy_measures;

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
save PanSim_args PanSim_args T N_Sim Np Rstr_Vars
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
Policy_measures = string(T(Idx_Closest,Rstr_Vars).Variables);

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
Policy_measures = string(T(Idx_Closest,Rstr_Vars).Variables);

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

%% Simulation: strict restrictions only after k > 40

load PanSim_args.mat

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

beta0 = min(T.Beta);
[~,Idx_Closest] = max(abs(T.Beta - beta0));
No_restriction = string(T(Idx_Closest,Rstr_Vars).Variables);
[~,Idx_Closest] = min(abs(T.Beta - beta0));
Make_restrictions = string(T(Idx_Closest,Rstr_Vars).Variables);

N_restriction = 40;

nx = 4;
x = zeros(nx,N_Sim+1);
beta = [ zeros(1,N_Sim) beta0 ];

Policy_measures = No_restriction;
for k = 2:N_Sim+1

    if k == N_restriction
        Policy_measures = Make_restrictions;
    end

    % Execute PanSim
    simout = obj.runForDay(Policy_measures);
    [simx,simbeta] = get_SEIRb(simout,Np);    

    x(:,k) = simx;
    beta(k-1) = simbeta;

end

save PanSim_max_after x beta beta0 Policy_measures

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

load PanSim_max_after.mat
x_max_after = x;
beta_max_after = beta;
pm_max_after = strjoin(strtrim(Policy_measures),', ') + " -- strictest restrictions after 40 days";

load PanSim_args.mat
t = 0:N_Sim;

fig = figure(13);
fig.Position(3:4) = [716 905];
Tl = tiledlayout(2,1,'Padding','compact','TileSpacing','tight');

ax = nexttile;
plot(t,[x_min(3,:);x_max(3,:);x_max_after(3,:)]'/Np*100)
Leg = legend(pm_min,pm_max,pm_max_after);
title 'All infected (percentage)'
xlabel days
ylabel 'Inf/Population * 100'
grid on

ylim([0 9])

ax = nexttile;
plot(t,[beta_min;beta_max;beta_max_after]')
Leg = legend(pm_min,pm_max,pm_max_after);
title 'Estimated beta'
xlabel days
ylabel 'beta'
grid on
