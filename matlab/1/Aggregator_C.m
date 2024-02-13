function [D,S] = Aggregator_C(Tp,args)
arguments
    Tp = 7;
    args.Window_Radius = 7;
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 07. (2023a)
%

Rstr_Vars = policy_varnames;

DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results";
Directories = dir(fullfile(DIR,"*_Finalized"));

xls = fullfile(DIR,Directories(1).name,"A.xls");
opts = detectImportOptions(xls);
opts = setvartype(opts,Rstr_Vars,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts.SelectedVariableNames = ["Date" Rstr_Vars "TrRate" "S" "L" "P" "I" "A" "H"];

xlsname = @(i) fullfile(DIR,Directories(i).name,"A.xls");
D = readtimetable(xlsname(1),opts);
for i = 2:length(Directories)
    Path = fullfile(DIR,Directories(i).name,"A.xls");
    Di = readtimetable(Path,opts);
    Di.TrRate = movmean(Di.TrRate,7);

    D = [ D ; Di ];
end
D(ismissing(D.TrRate),:) = [];
D(D.TrRate < 0.01,:) = [];
D(D.Date > datetime(2021,03,01),:) = [];

D = quantify_policy(D);

T = load_policy_measures;

Dt = join(D,T(:,[Rstr_Vars,"Idx","Beta","Intelligent"]),"Keys",Rstr_Vars);

%%

Dt(~Dt.Intelligent,:) = [];
T(~T.Intelligent,:) = [];

Dg = groupsummary(Dt,"Idx",["mean","std"],"TrRate");

figure(412);
Tl = tiledlayout(1,1);
ax = nexttile;
hold on, box on, grid on;
plot(Dt.Idx,Dt.TrRate,'o','MarkerSize',3,'DisplayName','Measurement')
errorbar(Dg.Idx,Dg.mean_TrRate,Dg.std_TrRate,'s','LineWidth',2,'DisplayName','Measurement statistics')
plot(T.Idx,T.Beta,'LineWidth',2,'DisplayName','LUT');

% for degree = 2
%     pfit = polyfit(Dt.Idx,Dt.TrRate,degree);
%     plot(T.Idx,polyval(pfit,T.Idx),'LineWidth',2,'DisplayName',"polyfit(**," +  num2str(degree) + ")")
% end

for degree = 1:2
    pfit = polyfit(Dt.Idx,Dt.TrRate ./ Dt.Beta,degree);
    plot(T.Idx,polyval(pfit,T.Idx) .* T.Beta,'LineWidth',2,'DisplayName',"polyfit(**," +  num2str(degree) + ")")
end

YData = (width(T.Iq):-1:0)/50;

[ii,qq] = meshgrid([T.Idx ; T.Idx(end)+1],YData);
Sf = surf(ii,qq,T.Iq([1:end,end],[1:end,end])','HandleVisibility','off');
Sf.FaceAlpha = 0.5;
Sf.EdgeAlpha = 0;
Plot_Colors
colormap(ax,[Color_5;Color_3;Color_2]);

xlim([T.Idx(1),T.Idx(end)+1])

xline(T.Idx(1):10:T.Idx(end),'HandleVisibility','off')

for i = 1:width(T.Iq)
    for Idx = T.Idx(1)+1:30:T.Idx(end)
        Tx = text(Idx,(YData(i)+YData(i+1))/2,Rstr_Vars(i));
    end
end

legend

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
