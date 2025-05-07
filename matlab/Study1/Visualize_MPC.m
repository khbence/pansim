%%
% Author: Peter Polcz (ppolcz@gmail.com) 
% Created on 2023. August 22. (2023a)
% 
function [fig,ret] = Visualize_MPC(R,Pred,Idx,args)
arguments
    R,Pred
    Idx = 0
    args.FigDim = [1916 985]
    args.FigNr = 13
end
%%


persistent Figure Link_XLim INITIALIZED
persistent Pl_Imsd Pl_Iprd Pl_Bmsd Pl_Bcmd Pl_Bprd Pl_Int
persistent Pl_TP Pl_PL Pl_CF Pl_SO Pl_QU Pl_MA Xl
persistent Sf_TP Sf_PL Sf_CF Sf_SO Sf_QU Sf_MA Sf_Iq

if isempty(INITIALIZED) || ~INITIALIZED || Idx == 0 || ~isvalid(Figure)

    Plot_Colors
    MyColorMap = [Color_5 ; Color_3 ; Color_2];

    Today = R.Date( min(max(1,Idx),height(R)) );
    Xline = @() xline(Today,'-','','LineWidth',2,'Color',Color_2,'HandleVisibility','off');
    Cnt(double('A')-1);

    Figure = figure(args.FigNr);
    Figure.Position([3 4]) = args.FigDim;
    Tl = tiledlayout(6,2,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");
    
    Ax = nexttile; hold on
    Pl_Ioff = plot(R.Date,R.Ioff);
    Xl = Xline();
    make_title('Estimated epidemic curve in 2020/21 (for comparison)')
    
    Ax = [Ax nexttile([2 1])]; hold on
    Pl_Imsd = plot(R.Date,R.I,'DisplayName','Simulated by PanSim');
    Pl_Iref = plot(R.Date,R.Iref,'DisplayName','Prescribed curve');
    Pl_Iprd = plot(Pred.Date,Pred.I,'DisplayName','Predicted by Comp-SLPIA');
    Leg = legend('Location','northeast','Interpreter','latex','FontSize',12);    
    Xl = [Xl Xline()];
    make_title('Infected')

    Ax = [Ax nexttile([2 1])]; hold on
    Pl_Int = plot_interval(R.Date,R.TrRateBounds(:,1),R.TrRateBounds(:,2));
    Pl_Bmsd = plot(R.Date,R.TrRate,'Color',Color_1,'DisplayName','Measured by PanSim');
    Pl_Bcmd = stairs(R.Date,R.TrRateCmd,'Color',Color_2,'LineWidth',1.5,'DisplayName','Prescribed by MPC');
    Pl_Bprd = stairs(R.Date,R.TrRateExp,'Color',Color_3,'DisplayName','Expected');
    Pl_Int(1).HandleVisibility = 'off';
    Pl_Int(2).HandleVisibility = 'off';
    Pl_Int(3).DisplayName = 'Possible range';
    Leg = legend('Location','northwest','Interpreter','latex','FontSize',12);
    Xl = [Xl Xline()];
    ylim([0,1])
    make_title('Transmission rate ($\beta$)')

    Ax = [Ax nexttile]; hold on
    Ax_NO_GRID = Ax(end);
    YData = 0:width(R.Iq);
    [DD,YY] = meshgrid(R.Date,YData);
    Sf_Iq = surf(DD,YY,R.Iq(:,[1:end,end])');
    Sf_Iq.EdgeAlpha = 0;
    Sf_Iq.FaceAlpha = 0.8;
    Yl = yline(YData,'k');
    colormap(Ax(end),MyColorMap)
    view(Ax(end),[0 90]);
    yticks(YData(2:end)-0.5);
    yticklabels(policy_varnames);
    Xl = [Xl Xline()];
    Xl(end).Color = [0,0,0];
    make_title('Intervention')
    view([0 -90])

    Ax = [Ax nexttile]; hold on
    Pl_TP = stairs(R.Date,R.TP_Val,'k','LineWidth',2);
    ylabel(TeX("\% of population"),Interpreter="latex")
    yticks([0.5,1.5,3.5]), ylim([0,4])
    Xl = [Xl Xline()];
    Sf_TP = surf([R.Date,R.Date],R.TP_Val*0 + Ax(end).YLim,[R.TP_Val,R.TP_Val]-10);
    Sf_TP.EdgeAlpha = 0;
    Sf_TP.FaceAlpha = 0.4;
    colormap(Ax(end),MyColorMap)
    make_title('[TP] Testing intensity (\% of population per day)')
    
    Ax = [Ax nexttile]; hold on
    Pl_PL = stairs(R.Date,R.PL_Val,'k','LineWidth',2);
    yticks([0,1]), ylim([-0.2,1.2]), yticklabels(["NO","YES"])
    Xl = [Xl Xline()];
    Sf_PL = surf([R.Date,R.Date],R.PL_Val*0 + Ax(end).YLim,[R.PL_Val,R.PL_Val]-10);
    Sf_PL.EdgeAlpha = 0;
    Sf_PL.FaceAlpha = 0.4;
    colormap(Ax(end),MyColorMap)
    make_title('[PL] Entertainment venues to close or not')
    % make_title('Szórakozó helyeket lezárása vagy sem')
    
    Ax = [Ax nexttile]; hold on
    Pl_CF = stairs(R.Date,R.CF_Val,'k','LineWidth',2);
    yticks([0,1]), ylim([-0.2,1.2]), yticklabels(["NO","YES"])
    Xl = [Xl Xline()];
    Sf_CF = surf([R.Date,R.Date],R.CF_Val*0 + Ax(end).YLim,[R.CF_Val,R.CF_Val]-10);
    Sf_CF.EdgeAlpha = 0;
    Sf_CF.FaceAlpha = 0.4;
    colormap(Ax(end),MyColorMap)
    make_title('[CF] Curfew between 8 pm and 5 am')
    % make_title('Kijárási korlátozás este 8 és reggel 5 között')
    
    Ax = [Ax nexttile]; hold on
    Pl_SO = stairs(R.Date,R.SO_Val,'k','LineWidth',2);
    yticks([0,1,2]), ylim([-0.4,2.4]), yticklabels(["NO","up to gr.3","up to gr.12"])
    Xl = [Xl Xline()];
    Sf_SO = surf([R.Date,R.Date],R.SO_Val*0 + Ax(end).YLim,[R.SO_Val,R.SO_Val]-10);
    Sf_SO.EdgeAlpha = 0;
    Sf_SO.FaceAlpha = 0.4;
    colormap(Ax(end),MyColorMap)
    make_title('[SC] School closures')
    % make_title('Iskolák zárása')
    
    Ax = [Ax nexttile]; hold on
    Pl_QU = stairs(R.Date,R.QU_Val,'k','LineWidth',2);
    yticks([0,2,4]), ylim([-0.4,4.4]), yticklabels(["NO",TeX("household"),"hh+wrkpl"])
    Xl = [Xl Xline()];
    Sf_QU = surf([R.Date,R.Date],R.QU_Val*0 + Ax(end).YLim,[R.QU_Val,R.QU_Val]-10);
    Sf_QU.EdgeAlpha = 0;
    Sf_QU.FaceAlpha = 0.4;
    colormap(Ax(end),MyColorMap)
    make_title('[QU] Strictness of the quarantine policy')
    % make_title('Karantén irányelv szigorúsága')
    
    Ax = [Ax nexttile]; hold on
    Pl_MA = stairs(R.Date,R.MA_Val,'k','LineWidth',2);
    yticks([0.8,1.0]), ylim([0.75,1.05])
    Xl = [Xl Xline()];
    Sf_MA = surf([R.Date,R.Date],R.MA_Val*0 + Ax(end).YLim,-[R.MA_Val,R.MA_Val]-1);
    Sf_MA.EdgeAlpha = 0;
    Sf_MA.FaceAlpha = 0.4;
    colormap(Ax(end),MyColorMap)
    make_title('[MA] Coefficient that slows the spread of the epidemic by masking')
    % make_title('A járvány terjedését maszkviseléssel lassító együttható')
    
    Link_XLim = linkprop(Ax,'XLim');
    
    XLim = [R.Date([1,end])];

    for ax = Ax
        grid(ax,'on')
        box(ax,'on')
    end
    
    for ax = Ax
        ax.TickLabelInterpreter = "latex";
        ax.FontSize = 12;
        
        ax.XTick = R.Date(day(R.Date) == 1 & mod(month(R.Date),2)==0);
        ax.XMinorGrid = 'on';
        ax.XAxis.MinorTick = 'off';
        ax.XAxis.MinorTickValues = R.Date(weekday(R.Date) == 1); 
        ax.XLim = XLim;
    end
    
    Ax_NO_GRID.YGrid = 'off';

    INITIALIZED = true;

    fig = Figure;
    ret.Link_XLim = Link_XLim;
    return
end

% Pl_Imsd.XData = R.Date;
Pl_Imsd.YData = R.I;
assert(numel(Pl_Imsd.XData) == numel(Pl_Imsd.YData))

Pl_Iprd.XData = Pred.Date;
Pl_Iprd.YData = Pred.I;

% Pl_Bmsd.XData = R.Date;
Pl_Bmsd.YData = R.TrRate;
assert(numel(Pl_Bmsd.XData) == numel(Pl_Bmsd.YData))

% Pl_Bcmd.XData = Pred.Date;
% Pl_Bcmd.YData = Pred.TrRate;
Pl_Bcmd.YData = R.TrRateCmd;

Pl_Int(1).YData = R.TrRateBounds(:,2);
Pl_Int(2).YData = R.TrRateBounds(:,1);
Pl_Int(3).YData = [ Pl_Int(1).YData(1) ; Pl_Int(1).YData' ; Pl_Int(1).YData(end) ; Pl_Int(2).YData(end:-1:1)' ];

% Pl_Bprd.XData = R.Date;
Pl_Bprd.YData = R.TrRateExp;
assert(numel(Pl_Bprd.XData) == numel(Pl_Bprd.YData))

Sf_Iq.ZData = R.Iq(:,[1:end,end])';

% Pl_TP.XData = R.Date;
Pl_TP.YData = R.TP_Val;
Sf_TP.ZData = [R.TP_Val,R.TP_Val]-10;
Sf_TP.CData = Sf_TP.ZData;
assert(numel(Pl_TP.XData) == numel(Pl_TP.YData))

% Pl_PL.XData = R.Date;
Pl_PL.YData = R.PL_Val;
Sf_PL.ZData = [R.PL_Val,R.PL_Val]-10;
Sf_PL.CData = Sf_PL.ZData;
assert(numel(Pl_PL.XData) == numel(Pl_PL.YData))

% Pl_CF.XData = R.Date;
Pl_CF.YData = R.CF_Val;
Sf_CF.ZData = [R.CF_Val,R.CF_Val]-10;
Sf_CF.CData = Sf_CF.ZData;
assert(numel(Pl_CF.XData) == numel(Pl_CF.YData))

% Pl_SO.XData = R.Date;
Pl_SO.YData = R.SO_Val;
Sf_SO.ZData = [R.SO_Val,R.SO_Val]-10;
Sf_SO.CData = Sf_SO.ZData;
assert(numel(Pl_SO.XData) == numel(Pl_SO.YData))

% Pl_QU.XData = R.Date;
Pl_QU.YData = R.QU_Val;
Sf_QU.ZData = [R.QU_Val,R.QU_Val]-10;
Sf_QU.CData = Sf_QU.ZData;
assert(numel(Pl_QU.XData) == numel(Pl_QU.YData))

% Pl_MA.XData = R.Date;
Pl_MA.YData = R.MA_Val;
Sf_MA.ZData = 1-[R.MA_Val,R.MA_Val]-10;
Sf_MA.CData = Sf_MA.ZData;
assert(numel(Pl_MA.XData) == numel(Pl_MA.YData))

for xl = Xl
    xl.Value = R.Date(Idx);
end

fig = Figure;

end

function make_title(str)
    title(TeX("\makebox[14.5cm]{\textbf{"+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end

function test
%%
    Visualize_MPC(R,Pred,1)
    Visualize_MPC(R,Pred,2)
    Visualize_MPC(R,Pred,3)
    Visualize_MPC(R,Pred,4)

%%
    R = readtimetable('/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_2024-02-02_15-49_T16_randref_Finalized/A.xls');
    Pred = R(:,["S" "L" "P" "I" "A" "TrRate"]);
    
    Visualize_MPC(R,Pred,160);

end

