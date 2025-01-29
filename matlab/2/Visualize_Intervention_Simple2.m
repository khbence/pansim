%%
% Author: Peter Polcz (ppolcz@gmail.com) 
% Created on 2024. February 07. (2023a)
% Modified on 2025. January 22. (2024b) -- PLOS CB minor review
% 
function [fig,ret] = Visualize_Intervention_Simple2(R,XData,args)
arguments
    R
    XData = (0:height(R)-1)'
    args.FigDim = [905 616]
    args.FigNr = 1315
    args.XLineData = [];
    args.TrRateVarName = "TrRate"
    args.TrRateStd = ''
    args.TrRateMedian = ''
end
%%

if isa(R,"timetable")
    XData = R.Properties.RowTimes;
end

XData = [ XData(:) ; XData(end)+1 ];
R(end+1,:) = R(end,:);

Plot_Colors
MyColorMap = [Color_5 ; Color_3 ; Color_2];

% Today = R.Date( min(max(1,Idx),height(R)) );
% Xline = @() xline(Today,'-','','LineWidth',2,'Color',Color_2,'HandleVisibility','off');
Cnt(double('A')-1);


fig = figure(args.FigNr);
fig.Position([3 4]) = args.FigDim;
Tl = tiledlayout(2,1,"TileSpacing","tight","Padding","compact","TileIndexing","columnmajor");


Ax = nexttile; hold on
if ~isempty(args.TrRateStd)
    Sh = plot_mean_var(XData,R.(args.TrRateVarName),R.(args.TrRateStd),[1,1,1]*0.5,'PlotLim',false,'Alpha',1);
else
    Pl_Tr = stairs(XData,R.(args.TrRateVarName),'k','LineWidth',2);
end
if ~isempty(args.TrRateMedian)
    Pl_Tr = plot(XData,R.(args.TrRateMedian),'LineWidth',2);
end
ylabel(TeX("tr. rate ($\beta$)"),Interpreter="latex")
make_title('Estimated transmission rate $\pm$\,std ($\approx 68\%$\,CI)')
% make_title('Estimated transmission rate $\pm 2$\,std ($\approx 95\%$\,CI)')

Ax = [Ax nexttile]; hold on
Ax_NO_GRID = Ax(end);
YData = 0:width(R.Iq);
[DD,YY] = meshgrid(XData,YData);
Idx = [5 6 2 3 4 1];
Sf_Iq = surf(DD,YY,R.Iq(:,[Idx,Idx(end)])');
Sf_Iq.EdgeAlpha = 0;
Sf_Iq.FaceAlpha = 0.8;
Yl = yline(YData,'k');
colormap(Ax(end),MyColorMap)
view(Ax(end),[0 90]);
yticks(YData(2:end)-0.5);
yticklabels(Vn.policy(Idx));
make_title('Intervention')
view([0 -90])

Link_XLim = linkprop(Ax,'XLim');

XLim = [XData(1) XData(end)];

for ax = Ax
    grid(ax,'on')
    box(ax,'on')
end

for ax = Ax
    ax.TickLabelInterpreter = "latex";
    ax.FontSize = 12;
    
    ax.XTick = XData(day(XData) == 1 & mod(month(XData),2)==0);
    ax.XMinorGrid = 'on';
    ax.XAxis.MinorTick = 'off';
    ax.XAxis.MinorTickValues = XData(weekday(XData) == 1); 
    ax.XLim = XLim;
end
Ax_NO_GRID.YGrid = 'off';

ret.Link_XLim = Link_XLim;

exportgraphics(fig,"/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/Dokumentaciok/Docs_CsutakB_PhD/11_Epid_MPC_Agent/actual/fig/Fig" + string(args.FigNr) + ".png")

end

function make_title(str)
    title(TeX("\makebox[14.5cm]{\textbf{"+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end
