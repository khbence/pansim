%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
% 
% `Idx` should be set first

return

%%

fp = pcz_mfilename(mfilename("fullpath"));

ff = @(d) string(cellfun(@(s) {fullfile(s.folder,s.name)}, num2cell(d)));
dirname = fullfile(fp.dir,"Output");
dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output";
xlsnames = ff( dir(fullfile(dirname,"Allcomb_*",'*.xls')) );

recdir = fullfile(dirname,"AllCombRec_2025-02-10");
if ~exist(recdir,"dir")
    mkdir(recdir)
end

opts = detectImportOptions(xlsnames(1));
opts = setvartype(opts,opts.SelectedVariableNames,"double");
opts = setvartype(opts,Vn.policy,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts = setvartype(opts,"IQ","int32");

T = readtimetable(xlsnames(1),opts,'Sheet','Results');

Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2024-02-26_Agens_Wild.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,T.Date(1),T.Date(end)),:);
P = hp.param2table(P.Param);

Nr_xls = length(xlsnames);

%%


N = height(T)*numel(xlsnames);

Z = zeros(height(T),numel(xlsnames));
I_all = Z;
Ir_all = Z;

VariableNames = ["NewCases","NewCasesRate","Rt","IQ","TrRateRec",Vn.policy + "_Val"];
na = repmat({nan(N,1)},size(VariableNames));
D = table(na{:},'VariableNames',VariableNames);

S = cell(1,Nr_xls);
Sr = cell(1,Nr_xls);
L = cell(1,Nr_xls);
Lr = cell(1,Nr_xls);
P = cell(1,Nr_xls);
Pr = cell(1,Nr_xls);
I = cell(1,Nr_xls);
Ir = cell(1,Nr_xls);
A = cell(1,Nr_xls);
Ar = cell(1,Nr_xls);
H = cell(1,Nr_xls);
Hr = cell(1,Nr_xls);

for i = 1:Nr_xls
    tic

    T = readtimetable(xlsnames(i),opts,'Sheet','Results');
    R = readtimetable(xlsnames(i),opts,'Sheet','Reconstruction');

    ldx = (T.L + T.P + T.I + T.A) ./ C.Np * 100000 >= 10;

    S{i} = T.S(ldx);
    Sr{i} = R.Sr(ldx);
    L{i} = T.L(ldx);
    Lr{i} = R.Lr(ldx);
    P{i} = T.P(ldx);
    Pr{i} = R.Pr(ldx);
    I{i} = T.I(ldx);
    Ir{i} = R.Ir(ldx);
    A{i} = T.A(ldx);
    Ar{i} = R.Ar(ldx);
    H{i} = T.H(ldx);
    Hr{i} = R.Hr(ldx);

    fprintf('%d/%d   ',i,Nr_xls);
    toc
end

save('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/RecUncAnal_2025-02-11.mat','S','Sr','L','Lr','P','Pr','I','Ir','A','Ar','H','Hr')

return

%%

load('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/RecUncAnal_2025-02-11.mat')

c = 1; % 00000 / C.Np;
r = 501;
method = 'mean';

SS = {};
[SS.Sim,Idx] = sort(vertcat(S{:})*c);
SS.Rec = At(vertcat(Sr{:})*c,Idx);
SS.Std = movstd(SS.Rec,r);
SS.Mean = movmean(SS.Rec,r);
[SS.MeanStd,SS.Val,SS.Cnt] = groupsummary([SS.Mean,SS.Std],SS.Sim,method);

LL = {};
[LL.Sim,Idx] = sort(vertcat(L{:})*c);
LL.Rec = At(vertcat(Lr{:})*c,Idx);
LL.Std = movstd(LL.Rec,r);
LL.Mean = movmean(LL.Rec,r);
[LL.MeanStd,LL.Val,LL.Cnt] = groupsummary([LL.Mean,LL.Std],LL.Sim,method);

PP = {};
[PP.Sim,Idx] = sort(vertcat(P{:})*c);
PP.Rec = At(vertcat(Pr{:})*c,Idx);
PP.Std = movstd(PP.Rec,r);
PP.Mean = movmean(PP.Rec,r);
[PP.MeanStd,PP.Val,PP.Cnt] = groupsummary([PP.Mean,PP.Std],PP.Sim,method);

AA = {};
[AA.Sim,Idx] = sort(vertcat(A{:})*c);
AA.Rec = At(vertcat(Ar{:})*c,Idx);
AA.Std = movstd(AA.Rec,r);
AA.Mean = movmean(AA.Rec,r);
[AA.MeanStd,AA.Val,AA.Cnt] = groupsummary([AA.Mean,AA.Std],AA.Sim,method);

II = {};
[II.Sim,Idx] = sort(vertcat(I{:})*c);
II.Rec = At(vertcat(Ir{:})*c,Idx);
II.Std = movstd(II.Rec,r);
II.Mean = movmean(II.Rec,r);
[II.MeanStd,II.Val,II.Cnt] = groupsummary([II.Mean,II.Std],II.Sim,method);

HH = {};
[HH.Sim,Idx] = sort(vertcat(H{:})*c);
HH.Rec = At(vertcat(Hr{:})*c,Idx);
HH.Std = movstd(HH.Rec,r);
HH.Mean = movmean(HH.Rec,r);
[HH.MeanStd,HH.Val,HH.Cnt] = groupsummary([HH.Mean,HH.Std],HH.Sim,method);

UU = {};
[UU.Sim,Idx] = sort(LL.Sim + PP.Sim + AA.Sim);
UU.Rec = At(LL.Rec + PP.Rec + AA.Rec,Idx);
UU.Std = movstd(UU.Rec,r);
UU.Mean = movmean(UU.Rec,r);
[UU.MeanStd,UU.Val,UU.Cnt] = groupsummary([UU.Mean,UU.Std],UU.Sim,method);

%%

FontSize = 14;
ILim = 1000;
a = 2;

fig = figure(31231);
fig.Position(3:4) = [795 1043];
Tl = tiledlayout(5,1,'TileSpacing','tight','Padding','compact');

kisR = 51;
Ax = nexttile;
XLims = [0,ILim];
yyaxis left
histogram(LL.Sim,'FaceAlpha',0.5,'BinWidth',1,'EdgeAlpha',0,'BinEdges',XLims(1):XLims(2));
yyaxis right
hold on
plot(LL.Val,movmean(LL.Val-LL.MeanStd(:,1),kisR))
plot(LL.Val,movmean(LL.Val-LL.MeanStd(:,1)+a*LL.MeanStd(:,2),kisR))
plot(LL.Val,movmean(LL.Val-LL.MeanStd(:,1)-a*LL.MeanStd(:,2),kisR),'--')
xlim(XLims);
xlabel('$\mathbf{L}$ (in a 180k population)','Interpreter','latex','FontSize',FontSize)
% title({ ...
%     sprintf('Distribution of the %d,%d samples w.r.t. to number of infected people (left axes)',floor(numel(LL.Sim)/1000),mod(numel(LL.Sim),1000))
%     'Mean (solid line) and the 95\%CI (dashed lines) of reconstruction error (right axes)'
%     },'Interpreter','latex','FontSize',FontSize)

Ax = [Ax nexttile];
XLims = [0,ILim];
yyaxis left
histogram(PP.Sim,'FaceAlpha',0.5,'BinWidth',1,'EdgeAlpha',0,'BinEdges',XLims(1):XLims(2));
yyaxis right
hold on
plot(PP.Val,movmean(PP.Val-PP.MeanStd(:,1),kisR))
plot(PP.Val,movmean(PP.Val-PP.MeanStd(:,1)+a*PP.MeanStd(:,2),kisR))
plot(PP.Val,movmean(PP.Val-PP.MeanStd(:,1)-a*PP.MeanStd(:,2),kisR),'--')
xlim(XLims);
xlabel('$\mathbf{P}$ (in a 180k population)','Interpreter','latex','FontSize',FontSize)

Ax = [Ax nexttile];
XLims = [0,ILim];
yyaxis left
histogram(AA.Sim,'FaceAlpha',0.5,'BinWidth',1,'EdgeAlpha',0,'BinEdges',XLims(1):XLims(2));
yyaxis right
hold on
plot(AA.Val,movmean(AA.Val-AA.MeanStd(:,1),kisR))
plot(AA.Val,movmean(AA.Val-AA.MeanStd(:,1)+a*AA.MeanStd(:,2),kisR))
plot(AA.Val,movmean(AA.Val-AA.MeanStd(:,1)-a*AA.MeanStd(:,2),kisR),'--')
xlim(XLims);
xlabel('$\mathbf{A}$ (in a 180k population)','Interpreter','latex','FontSize',FontSize)

Ax = [Ax nexttile];
XLims = [0,ILim];
yyaxis left
histogram(II.Sim,'FaceAlpha',0.5,'BinWidth',1,'EdgeAlpha',0,'BinEdges',XLims(1):XLims(2));
yyaxis right
hold on
plot(II.Val,movmean(II.Val-II.MeanStd(:,1),kisR))
plot(II.Val,movmean(II.Val-II.MeanStd(:,1)+a*II.MeanStd(:,2),kisR))
plot(II.Val,movmean(II.Val-II.MeanStd(:,1)-a*II.MeanStd(:,2),kisR),'--')
xlim(XLims);
xlabel('$\mathbf{I}$ (in a 180k population)','Interpreter','latex','FontSize',FontSize)

% Ax = [Ax nexttile];
% XLims = [0,ILim*3];
% yyaxis left
% histogram(movmean(UU.Sim,1001),'FaceAlpha',0.5,'BinWidth',1,'EdgeAlpha',0,'BinEdges',XLims(1):XLims(2));
% yyaxis right
% hold on
% plot(UU.Val,movmean(UU.Val-UU.MeanStd(:,1),kisR))
% plot(UU.Val,movmean(UU.Val-UU.MeanStd(:,1)+a*UU.MeanStd(:,2),kisR))
% plot(UU.Val,movmean(UU.Val-UU.MeanStd(:,1)-a*UU.MeanStd(:,2),kisR),'--')
% xlim(XLims);
% xlabel('$\mathbf{L}+\mathbf{P}+\mathbf{A}$ (in a 180k population)','Interpreter','latex','FontSize',FontSize)

Ax = [Ax nexttile];
XLims = [C.Np-100000,C.Np];
yyaxis left
histogram(movmean(SS.Sim,1001),'FaceAlpha',0.5,'BinWidth',1,'EdgeAlpha',0,'BinEdges',XLims(1):XLims(2));
yyaxis right
hold on
plot(SS.Val,movmean(SS.Val-SS.MeanStd(:,1),kisR))
plot(SS.Val,movmean(SS.Val-SS.MeanStd(:,1)+a*SS.MeanStd(:,2),kisR))
plot(SS.Val,movmean(SS.Val-SS.MeanStd(:,1)-a*SS.MeanStd(:,2),kisR),'--')
xlim(XLims);
xlabel('$\mathbf{S}$ (in a 180k population)','Interpreter','latex','FontSize',FontSize)

idx = 0;
for ax = Ax
    idx = idx + 1;
    ax.FontSize = FontSize;
    ax.TickLabelInterpreter = 'latex';
    grid(ax,'on')
    box(ax,'on')

    ax.XTick = unique([ax.XLim(1) ax.XTick ax.XLim(2)]);
    ax.XTick( find(diff(ax.XTick) ./ (ax.XLim(2) - ax.XLim(1)) < 0.1) + 1 ) = [];
    
    ax.YAxis(1).Label.String = 'No. samples';
    ax.YAxis(1).Label.Interpreter = 'latex';
    ax.YAxis(1).Label.FontSize = FontSize;
    
    ax.YAxis(2).Label.String = 'Bias$\,\pm\,2\,$Std';
    ax.YAxis(2).Label.Interpreter = 'latex';
    ax.YAxis(2).Label.FontSize = FontSize;
    

    % if mod(idx,2) == 1
        % ylabel(ax,'Samples','Interpreter','latex','FontSize',FontSize)
    % end
end

exportgraphics(fig,"/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/Dokumentaciok/Docs_CsutakB_PhD/11_Epid_MPC_Agent/actual/fig_All/" + ...
    "Distribution_of_Samples.png")

%%

Plot_Colors
MyColorMap = [Color_5 ; Color_3 ; Color_2];

a = 2;
% for i = 1:Nr_xls
for i = [19,48,64,78,84,101,155,158,133,202,266,273,276,299,229,304,322,327,336,345,350,354,482,483,541,580,403,966]
    tic

    T = readtimetable(xlsnames(i),opts,'Sheet','Results');
    R = readtimetable(xlsnames(i),opts,'Sheet','Reconstruction');

    Iq = T(:,Vn.policy_Iq_);
    T.Iq = Iq.Variables;

    T.I(T.I/C.Np*1e5 < 10) = NaN;

    % mI = max(T.I)
    % if mI < 1500 || 1700 < mI
    %     continue
    % end

    Lr_Mean = interp1(LL.Val,LL.MeanStd(:,1),T.L);
    Lr_Std = interp1(LL.Val,LL.MeanStd(:,2),T.L);

    Ir_Mean = interp1(II.Val,II.MeanStd(:,1),T.I);
    Ir_Std = interp1(II.Val,II.MeanStd(:,2),T.I);

    Ur_Mean = interp1(UU.Val,UU.MeanStd(:,1),T.L+T.P+T.A);
    Ur_Std = interp1(UU.Val,UU.MeanStd(:,2),T.L+T.P+T.A);

    Sr_Mean = interp1(SS.Val,SS.MeanStd(:,1),T.S);
    Sr_Std = interp1(SS.Val,SS.MeanStd(:,2),T.S);

    Idx_first_missing = find(ismissing(Lr_Mean+Ir_Mean+Ur_Mean+Sr_Mean),1);
    if isempty(Idx_first_missing)
        Idx_first_missing = height(T);
    end
    if Idx_first_missing < 10
        continue
    end
    Day = 1:Idx_first_missing;

    %%
    fig = figure(123);
    fig.Position(3:4) = [474 1050];
    Tl = tiledlayout(4,1,"TileSpacing","compact","Padding","compact");


    Ax = nexttile;
    hold on; grid on; box on;
    Plmv = plot_mean_var(Day,R.Sr(Day),Sr_Std(Day),'Alpha',a,'LineStyle','-','PlotLim',false);
    Pls = plot(Day,T.S(Day),'LineWidth',1,'Color',Color_2);
    Plr = plot(Day,R.Sr(Day),'LineWidth',1,'Color',Color_1);
    PlD = plot(Day(1),0,'-','Color','white');
    Leg = legend([Pls,Plmv(4)],{'Simulated quantities ($\mathbf{S},\mathbf{L},\mathbf{I}$)','Reconstructed (mean$\,\pm \,$2\,std)'},'Interpreter','latex');
    Leg.FontSize = FontSize;
    Leg.Location = "northoutside";
    Leg.NumColumns = 1;
    Leg.Box = 'off';
    Ax.FontSize = FontSize;
    Ax.TickLabelInterpreter = "latex";
    % xlabel('Days','Interpreter','latex')
    axis tight
    % Ax.YAxis.Exponent = 0;
    
    Ax = nexttile;
    hold on; grid on; box on;
    Plmv = plot_mean_var(Day,R.Lr(Day),Lr_Std(Day),'Alpha',a,'LineStyle','-','PlotLim',false);
    Pls = plot(Day,T.L(Day),'LineWidth',1,'Color',Color_2);
    Plr = plot(Day,R.Lr(Day),'LineWidth',1,'Color',Color_1);
    PlD = plot(Day(1),0,'-','Color','white');
    % ax1.XTickLabel = {};
    Ax.FontSize = FontSize;
    Ax.TickLabelInterpreter = "latex";
    ylabel('~~~~~~~~ Number of latent ($\mathbf{L}$), ~~~~~~~~ infected ($\mathbf{I}$), ~~~ susceptibles ($\mathbf{S}$) in a 180k population','Interpreter','latex','FontSize',FontSize)
    axis tight

    Ax = nexttile;
    hold on; grid on; box on;
    Plmv = plot_mean_var(Day,R.Ir(Day),Ir_Std(Day),'Alpha',a,'LineStyle','-','PlotLim',false);
    Pls = plot(Day,T.I(Day),'LineWidth',1,'Color',Color_2);
    Plr = plot(Day,R.Ir(Day),'LineWidth',1,'Color',Color_1);
    PlD = plot(Day(1),0,'-','Color','white');
    % ax1.XTickLabel = {};
    % Leg = legend([Pls,Plmv(4),PlD],{'Simulated $\mathbf{I}$','Reconstructed','(mean$\,\pm \,95\%\,$CI)'},'Interpreter','latex');
    % Leg.FontSize = FontSize;
    % Leg.Location = "best";
    Ax.FontSize = FontSize;
    Ax.TickLabelInterpreter = "latex";
    axis tight


    AxS = nexttile; hold on
    YData = 0:width(T.Iq);
    % --
    [DD,YY] = meshgrid([-4 -3,-2,-1],YData);
    surf(DD,YY,zeros(width(T.Iq)+1,1) + [0 0.5 1 1])
    % --
    [DD,YY] = meshgrid(Day,YData);
    Sf_Iq = surf(DD,YY,T.Iq(Day,[1:end,end])');
    Sf_Iq.EdgeAlpha = 0;
    Sf_Iq.FaceAlpha = 0.8;
    Yl = yline(YData,'k');
    colormap(AxS,MyColorMap)
    view(AxS,[0 90]);
    yticks(YData(2:end)-0.5);
    yticklabels(Vn.policy);
    xline([1,find(abs(diff(T.IQ)) > 0)'+1],'k','HandleVisibility','off')
    view([0 -90])
    AxS.XLim = Ax.XLim;
    AxS.FontSize = FontSize;
    AxS.TickLabelInterpreter = "latex";
    xlabel('Days','Interpreter','latex')
    ylabel('Applied interventions','Interpreter','latex','FontSize',FontSize)
    

    exportgraphics(fig,"/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/Dokumentaciok/Docs_CsutakB_PhD/11_Epid_MPC_Agent/actual/fig_All/" + ...
        "Reconstruction_Error" + num2str(i) + ".png")
    
    %%
    % keyboard

    fprintf('%d/%d   ',i,Nr_xls);
    toc
end
