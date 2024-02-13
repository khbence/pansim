function [D,Db] = Aggregator_E(T)
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 07. (2023a)
%

DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results";
Directories = dir(fullfile(DIR,"*_Finalized"));

xls = fullfile(DIR,Directories(1).name,"A.xls");
opts = detectImportOptions(xls);
opts = setvartype(opts,policy_varnames,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts.SelectedVariableNames = ["Date" policy_varnames simout_varnames "TrRate"];

xlsname = @(i) fullfile(DIR,Directories(i).name,"A.xls");
D = readtimetable(xlsname(1),opts);
for i = 2:length(Directories)
    Path = fullfile(DIR,Directories(i).name,"A.xls");
    Di = readtimetable(Path,opts);

    D = [ D ; Di ];
end
D(ismissing(D.TrRate),:) = [];
D(D.TrRate < 0.01,:) = [];

% Recompute the transmission rate
[~,D.TrRate] = get_SLPIAb(D(:,simout_varnames).Variables,'ImmuneIdx',2);

% D = quantify_policy(D);
D = join(D,T(:,[policy_varnames , "Idx" "Intelligent" "Beta"]));
D(~D.Intelligent,:) = [];

Start_Date = min(D.Date);
Days = days(D.Date - Start_Date) + randn(size(D.Date))*0.1;
[~,idx] = sort(Days);
D = D(idx,["Idx","TrRate"]);

isout = isoutlier(D.TrRate,"movmedian",31,'SamplePoints',Days(idx));
D(isout,:) = [];

Indices = sort(unique(D.Idx));

%%

Q = readtable("matlab/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");

P = Epid_Par.Get(Q);
Q("Future",:) = [];

P(P.Date < Start_Date,:) = [];
P.Beta0 = P.Pattern * Q.beta0;

D = join(D,P(:,"Beta0"));
D.Beta0 = D.Beta0 * 0.7;

B = retime(D(:,"Beta0"),"daily");
t = days(B.Date - Start_Date);


%%

fig = figure(512);
Tl = tiledlayout(5,5,"TileSpacing","tight","Padding","tight");

Color_1 = [0 0.4470 0.7410];
Color_2 = [0.8500 0.3250 0.0980];
Color_3 = [0.9290 0.6940 0.1250];
Color_4 = [0.4940 0.1840 0.5560];
Color_5 = [0.4660 0.6740 0.1880];
Color_6 = [0.3010 0.7450 0.9330];
Color_7 = [0.6350 0.0780 0.1840];

% Nr_Tune = 250;
% if numel(x) > Nr_Tune
%     idx = randperm(numel(x),Nr_Tune);
%     x_ = x(idx);
%     y_ = y(idx);
% else
%     x_ = x;
%     y_ = y;
% end
% hyp_init = struct('mean',[],'cov',[3,-0.5],'lik',-2);
% gpml_minimize = fn_gpml_minimize;
% hyp_tuned = gpml_minimize(hyp_init,@gp,-200,@infGaussLik,[],@covSEard,@likGauss,x_,y_);

Ax1 = [];
for Idx = Indices'
    ldx = D.Idx == Idx;

    if sum(ldx) < 100
        continue
    end




    x = days(D.Date(ldx) - Start_Date);
    y = D.TrRate(ldx) - D.Beta0(ldx);

    hyp = GP_hyp(struct('mean',[],'cov',[3,-0.5],'lik',-2),x,y);
    hyp.ell = 30;
    hyp.sf = 1.5;
    hyp.sn = 0.02;
    
    GP_eval(hyp);
    [GP_mean,GP_var] = GP_eval(hyp,t);


    if isempty(Ax1)
        Ax1 = nexttile;
        hold on, grid on, box on;
    end
    Good_mean = GP_mean + B.Beta0;
    % Good_mean(Good_mean - sqrt(GP_var) < 0 | sqrt(GP_var) > 0.2) = NaN;
    plot(Ax1,B.Date,Good_mean);
    
    ax = nexttile;
    hold on, grid on, box on
    plot(x + Start_Date,y + D.Beta0(ldx),'.k');
    plot_mean_var(B.Date,GP_mean+B.Beta0,sqrt(GP_var),Color_2);
    ylim([0,1]);

    title(join(string(T(T.Idx == Idx,policy_varnames).Variables),", "))
    
    fprintf("%d / %d\n", Idx, Indices(end));
end

end
