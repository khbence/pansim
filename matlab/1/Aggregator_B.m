function [D,S] = Aggregator_B(Rstr_Vars,Tp,args)
arguments
    Rstr_Vars;
    Tp = 7;
    args.Window_Radius = 7;
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 07. (2023a)
%

DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results";
Directories = dir(fullfile(DIR,"*_Finalized"));

fp = pcz_mfilename(mfilename("fullpath"));
Aggregator_DIR = fullfile(DIR,fp.bname);

if ~exist(Aggregator_DIR,"dir")
    mkdir(Aggregator_DIR)
end

xls = fullfile(DIR,Directories(1).name,"A.xls");
opts = detectImportOptions(xls);
opts = setvartype(opts,Rstr_Vars,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts.SelectedVariableNames = ["Date" Rstr_Vars "TrRate"];

xlsname = @(i) fullfile(DIR,Directories(i).name,"A.xls");
D = readtimetable(xlsname(1),opts);
for i = 2:length(Directories)
    Path = fullfile(DIR,Directories(i).name,"A.xls");
    Di = readtimetable(Path,opts);

    D = [ D ; Di ];
end
D(ismissing(D.TrRate),:) = [];
D(D.TrRate < 0.01,:) = [];

%%

d_sim = D.Date(1):Tp:D.Date(end);

if days(D.Date(end) - d_sim(end)) > 0
    d_sim = [ d_sim , D.Date(end) ];
end

N = numel(d_sim)-1;

Date1 = d_sim(1:end-1)';
Date2 = d_sim(1:end-1)';
min_TrRate = zeros(N,1) - 1e10;
max_TrRate = zeros(N,1) + 1e10;
mean_TrRate = nan(N,1);
std_TrRate = nan(N,1);

for i = 1:N
    Date1(i) = d_sim(i) - args.Window_Radius; 
    Date2(i) = d_sim(i+1)-1 + args.Window_Radius;
    Di = D(isbetween(D.Date,Date1(i),Date2(i)),:);
    Ds = groupsummary(Di,Rstr_Vars,["min","max","mean","std"],"TrRate");
    min_TrRate(i) = min(Ds.mean_TrRate);
    max_TrRate(i) = max(Ds.mean_TrRate);
    mean_TrRate(i) = mean(Di.TrRate);
    std_TrRate(i) = std(Di.TrRate);
end

Date = d_sim(1:end-1)';
S = timetable(Date,Date1,Date2,min_TrRate,max_TrRate,mean_TrRate,std_TrRate);

end
