%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%

Np = 178746;

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

%% Find closest policy measures

Beta_MinMax = [
    min(T.Beta) 
    max(T.Beta) 
    ];

beta_min = min(T.Beta);
beta_max = max(T.Beta);

lambda = 0.8;
queryBeta = [1-lambda lambda] * Beta_MinMax;

[~,Idx_Closest] = min(abs(T.Beta - queryBeta));
Policy_measures = string(T(Idx_Closest,1:end-1).Variables);

%% Reference signal

Mtp = 4;
Shift = 25;

N = 90; % days
t_sim = 0:N;
t_interp = t_sim + Shift;

REF_FILE = load("matlab/res2.mat");
x_ref = [1:length(REF_FILE.Ihat_daily); REF_FILE.Ihat_daily*Np*Mtp]';

spline_Iref = spline(x_ref(:, 1), x_ref(:, 2)); %create the funcion of the reference curve
spline_Irefd = fnder(spline_Iref, 1);           %calculate the first derivate
spline_Irefdd = fnder(spline_Iref, 2);          %calculate the second derivate

Iref = ppval(spline_Iref, t_interp).';
Irefd = ppval(spline_Irefd, t_interp).';
Irefdd = ppval(spline_Irefdd, t_interp).';

%% Compute beta

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
inp_params = ["panSim", "-r", " ", "--quarantinePolicy", "0", "-k", "0.00041", ...
     "--progression", "inputConfigFiles/progressions_Jun17_tune/transition_config.json", ...
     "-A", "inputConfigFiles/agentTypes_3.json", "-a", "inputRealExample/agents1.json", "-l", "inputRealExample/locations0.json", ...
     "--infectiousnessMultiplier", "0.98,1.81,2.11,2.58,4.32,6.8,6.8", "--diseaseProgressionScaling", "0.94,1.03,0.813,0.72,0.57,0.463,0.45"];
obj.initSimulation(inp_params);

k2 = 0.37;
k3 = 0.1429;

p_fp = [-0.1; -0.12; -0.14]*3.39322177189533; % *2
%p_fp = [-0.1; -0.12; -0.14]*1.25;
% p_fp = [-0.15 -0.1 -0.05] * 1.5;
feedback_pars = -place([0 1 0; 0 0 1; 0 0 0],[0; 0; 1],p_fp);
%feedback_pars = -lqr(A, B, 0.1*eye(3), 0.01);

c1 = feedback_pars(1);
c2 = feedback_pars(2);
c3 = feedback_pars(3);

Int_e = 0;

nx = 4;
x = zeros(nx,N+1);
beta_cmp = [ queryBeta zeros(1,N) ];
beta = [ queryBeta zeros(1,N) ];
dI = zeros(1,N+1);
PM = repmat(Policy_measures,[N+1,1]);

Polyfit_ord = 3;
Diff_N = 10;
Polyfit_t = -Diff_N+1:0;

for k = 2:N+1

    % Execute PanSim
    simout = obj.runForDay(Policy_measures);
    [simx,simbeta] = get_SEIRb(simout,Np);    

    x(:,k) = simx;

    S = simx(1);
    E = simx(2);
    I = simx(3);
    R = simx(4);

    e = Iref(k) - I;
    Int_e = Int_e + e;

    if k >= Diff_N+1
        Polyfit_I = x(3,Polyfit_t+k);
        p = polyfit(Polyfit_t,Polyfit_I,Polyfit_ord);
        dI(k) = p(end-1); % polyval(polyder(p),0)
    end

    de = Irefd(k) - dI(k);

    v = c1 * Int_e + c2 * e + c3 * de;
    u = (Irefdd(k) + E*(k2^2 + k3*k2) - I*k3^2 - v) / (I*S*k2) * Np;

    beta_cmp(k) = u;
    beta(k) = min(max(beta_min,u),beta_max);
    [~,Idx_Closest] = min(abs(T.Beta - beta(k)));
    Policy_measures = string(T(Idx_Closest,1:end-1).Variables);

end


% clear all
