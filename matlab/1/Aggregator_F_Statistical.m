function [T,d,Dly_Range,Dly_Count,Dly_Mean,Dly_Std,Dly_Median] = Aggregator_F_Statistical(TrRate_IDX)
arguments
    TrRate_IDX = 2
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Revised on 2024. February 07. (2023a)
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
[~,D.TrRate] = get_SLPIAb(D(:,simout_varnames).Variables,'ImmuneIdx',TrRate_IDX);

% D = quantify_policy(D);
T = load_policy_measures;
D = join(D,T(:,[policy_varnames , "Pmx" "Beta"]));
D(ismissing(D.Pmx),:) = [];
T(ismissing(T.Pmx),:) = [];
assert(all(diff(T.Pmx) == 1),'Policy records in T are not sorted')

Start_Date = min(D.Date);
End_Date = max(D.Date);
D.Days = days(D.Date - Start_Date);
[~,idx] = sort(D.Days);
D = D(idx,["Days","Pmx","TrRate"]);

isout = isoutlier(D.TrRate,"movmedian",31,'SamplePoints',sort(D.Days + randn(size(D.Days))*1e-5));
D(isout,:) = [];

N = days(End_Date - Start_Date) + 1;
nPM = max(T.Pmx);
Dly_Range = nan(N,2);
Dly_Count = zeros(N,nPM);
Dly_Median = nan(N,nPM);
Dly_Mean = nan(N,nPM);
Dly_Std = nan(N,nPM);

GS = groupcounts(D(:,"Pmx"),"Pmx");
PM_Count = zeros(1,nPM);
PM_Count(GS.Pmx) = GS.GroupCount;

idx_next_day = [0 ; find(diff(D.Days)) ; height(D)];
assert(numel(idx_next_day) == N+1);

%%

r = 3;
r = 10;
r = 15;
fprintf('Group summary\n');
for i = 1:N
    idx = idx_next_day(max(1,i-r))+1:idx_next_day(min(i+r+1,N+1));
    GS = groupsummary(D(idx,["Pmx","TrRate"]),"Pmx",["mean","median","std"]);
    Dly_Count(i,GS.Pmx) = GS.GroupCount;
    Dly_Median(i,GS.Pmx) = GS.median_TrRate;
    Dly_Mean(i,GS.Pmx) = GS.mean_TrRate;
    Dly_Std(i,GS.Pmx) = GS.std_TrRate;

    Dly_Range(i,:) = [ min(GS.mean_TrRate) , max(GS.mean_TrRate) ];

    fprintf('.');
    if mod(i,30) == 0
        fprintf('\n');
    end
    % fprintf('Group summary %d/%d\n',i,N);
end
fprintf('\n');


%%

Min_Count = 100;
Nr_Plots = sum(PM_Count >= Min_Count) + 1 + 1;

Tl_Rows = 1;
Tl_Cols = 1;
while Nr_Plots > Tl_Rows*Tl_Cols
    Tl_Rows = Tl_Rows+1;
    if Nr_Plots <= Tl_Rows*Tl_Cols
        break
    end
    Tl_Cols = Tl_Cols+1;
end


fig = figure(512);
Tl = tiledlayout(Tl_Rows,Tl_Cols,"TileSpacing","tight","Padding","tight");

Color_1 = [0 0.4470 0.7410];
Color_2 = [0.8500 0.3250 0.0980];
Color_3 = [0.9290 0.6940 0.1250];
Color_4 = [0.4940 0.1840 0.5560];
Color_5 = [0.4660 0.6740 0.1880];
Color_6 = [0.3010 0.7450 0.9330];
Color_7 = [0.6350 0.0780 0.1840];

d = Start_Date:End_Date;


for Pmx = unique(D.Pmx)'

    if PM_Count(Pmx) < Min_Count
        continue
    end

    ldx = ~isnan(Dly_Mean(:,Pmx));
    Mean = Dly_Mean(:,Pmx);
    Std = Dly_Std(:,Pmx);
    Median = Dly_Median(:,Pmx);

    ax = nexttile;
    hold on, grid on, box on
    plot_mean_var(d,Mean,Std,Color_2);
    plot(d,Median,'Color',Color_1);
    plot(d,Mean+2*Std,'Color',Color_3);
    plot(d,Mean-2*Std,'Color',Color_3);
    ylim([0,1]);

    Tl = title(string(num2str(Pmx)) + ": " + join(string(T(T.Pmx == Pmx,policy_varnames).Variables),", "));
    Tl.FontWeight = 'normal';
end

nexttile, hold on, grid on, box on;
plot_interval(d,Dly_Range(:,1),Dly_Range(:,2));
plot(d,Dly_Mean,'k')
ylim([0,1]);

nexttile, hold on, grid on, box on;
plot_interval(d,Dly_Range(:,1),Dly_Range(:,2));
ylim([0,1]);

end