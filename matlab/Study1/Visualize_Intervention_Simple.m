%%
% Author: Peter Polcz (ppolcz@gmail.com) 
% Created on 2024. February 07. (2023a)
% 
function [fig,ret] = Visualize_Intervention_Simple(R,XData,args)
arguments
    R
    XData = (0:height(R)-1)'
    args.FigDim = [1916 985]
    args.FigNr = 1315
    args.XLineData = [];
    args.TrRateVarName = "TrRate"
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
Tl = tiledlayout(8,1,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");


Ax = nexttile; hold on
Pl_Tr = stairs(XData,R.(args.TrRateVarName),'k','LineWidth',2)
ylabel(TeX("tr. rate ($\beta$)"),Interpreter="latex")
make_title('Estimated transmission rate')

Ax = [Ax nexttile]; hold on
Ax_NO_GRID = Ax(end);
YData = 0:width(R.Iq);
[DD,YY] = meshgrid(XData,YData);
Sf_Iq = surf(DD,YY,R.Iq(:,[1:end,end])');
Sf_Iq.EdgeAlpha = 0;
Sf_Iq.FaceAlpha = 0.8;
Yl = yline(YData,'k');
colormap(Ax(end),MyColorMap)
view(Ax(end),[0 90]);
yticks(YData(2:end)-0.5);
yticklabels(policy_varnames);
make_title('Intervention')
view([0 -90])

Ax = [Ax nexttile]; hold on
Pl_TP = stairs(XData,R.TP_Val,'k','LineWidth',2);
ylabel(TeX("\% of pop."),Interpreter="latex")
yticks([0.5,1.5,3.5]), ylim([0,4])
% Xl = [Xl Xline()];
Sf_TP = surf([XData,XData],R.TP_Val*0 + Ax(end).YLim,[R.TP_Val,R.TP_Val]-10);
Sf_TP.EdgeAlpha = 0;
Sf_TP.FaceAlpha = 0.4;
colormap(Ax(end),MyColorMap)
make_title('[TP] Testing intensity (\% of population per day)')

Ax = [Ax nexttile]; hold on
Pl_PL = stairs(XData,R.PL_Val,'k','LineWidth',2);
yticks([0,1]), ylim([-0.2,1.2]), yticklabels(["NO","YES"])
% Xl = [Xl Xline()];
Sf_PL = surf([XData,XData],R.PL_Val*0 + Ax(end).YLim,[R.PL_Val,R.PL_Val]-10);
Sf_PL.EdgeAlpha = 0;
Sf_PL.FaceAlpha = 0.4;
colormap(Ax(end),MyColorMap)
make_title('[PL] Entertainment venues to close or not')
% make_title('Szórakozó helyeket lezárása vagy sem')

Ax = [Ax nexttile]; hold on
Pl_CF = stairs(XData,R.CF_Val,'k','LineWidth',2);
yticks([0,1]), ylim([-0.2,1.2]), yticklabels(["NO","YES"])
% Xl = [Xl Xline()];
Sf_CF = surf([XData,XData],R.CF_Val*0 + Ax(end).YLim,[R.CF_Val,R.CF_Val]-10);
Sf_CF.EdgeAlpha = 0;
Sf_CF.FaceAlpha = 0.4;
colormap(Ax(end),MyColorMap)
make_title('[CF] Curfew between 8 pm and 5 am')
% make_title('Kijárási korlátozás este 8 és reggel 5 között')

Ax = [Ax nexttile]; hold on
Pl_SO = stairs(XData,R.SO_Val,'k','LineWidth',2);
yticks([0,1,2]), ylim([-0.4,2.4]), yticklabels(["NO","up to gr.3","up to gr.12"])
% Xl = [Xl Xline()];
Sf_SO = surf([XData,XData],R.SO_Val*0 + Ax(end).YLim,[R.SO_Val,R.SO_Val]-10);
Sf_SO.EdgeAlpha = 0;
Sf_SO.FaceAlpha = 0.4;
colormap(Ax(end),MyColorMap)
make_title('[SO] School closures')
% make_title('Iskolák zárása')

Ax = [Ax nexttile]; hold on
Pl_QU = stairs(XData,R.QU_Val,'k','LineWidth',2);
yticks([0,2,4]), ylim([-0.4,4.4]), yticklabels(["NO",TeX("household"),"hh+wrkpl"])
% Xl = [Xl Xline()];
Sf_QU = surf([XData,XData],R.QU_Val*0 + Ax(end).YLim,[R.QU_Val,R.QU_Val]-10);
Sf_QU.EdgeAlpha = 0;
Sf_QU.FaceAlpha = 0.4;
colormap(Ax(end),MyColorMap)
make_title('[QU] Strictness of the quarantine policy')
% make_title('Karantén irányelv szigorúsága')

Ax = [Ax nexttile]; hold on
Pl_MA = stairs(XData,R.MA_Val,'k','LineWidth',2);
yticks([0.8,1.0]), ylim([0.75,1.05])
% Xl = [Xl Xline()];
Sf_MA = surf([XData,XData],R.MA_Val*0 + Ax(end).YLim,-[R.MA_Val,R.MA_Val]-1);
Sf_MA.EdgeAlpha = 0;
Sf_MA.FaceAlpha = 0.4;
colormap(Ax(end),MyColorMap)
make_title('[MA] Coefficient that slows the spread of the epidemic by masking')
% make_title('A járvány terjedését maszkviseléssel lassító együttható')

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

end

function make_title(str)
    title(TeX("\makebox[14.5cm]{\textbf{"+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end
