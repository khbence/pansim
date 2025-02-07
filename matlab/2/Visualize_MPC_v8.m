%%
% Author: Peter Polcz (ppolcz@gmail.com) 
% Created on 2023. August 22. (2023a)
% 
function [fig,ret] = Visualize_MPC_v8(R,Idx,k,args)
arguments
    R
    Idx = 0
    k = 0
    args.Tp = 7
    args.FigDim = [590 704]
    args.FigNr = 13
    args.Reset = true
    args.BetaRange = [0 0.7]
end
%%

persistent Figure Link_XLim INITIALIZED
persistent Pl_Imsd Pl_Iprd Pl_Irec Pl_Bmsd Pl_BMmd Pl_Bcmd Pl_Bprd Pl_Brec Pl_BMrc Pl_Int Pl_Rnt
persistent Xl
persistent Sf_Iq

assert(mod(height(R)-1,args.Tp) == 0);

indices = reshape(2:height(R),args.Tp,[]);

MeanTrRate = repmat(mean(R.TrRate(indices),1),[args.Tp,1]);
MeanTrRate = MeanTrRate(:);
MeanTrRate = [MeanTrRate ; MeanTrRate(end)];

% MeanTrRateRec = repmat(mean(R.TrRateRec(indices),1),[args.Tp,1]);
% MeanTrRateRec = MeanTrRateRec(:);
% MeanTrRateRec = [MeanTrRateRec ; MeanTrRateRec(end)];

if ismember(Vn.Ipredk(k),R.Properties.VariableNames)
    Ipred = R.(Vn.Ipredk(k));
else
    Ipred = nan(height(R),1);
end

s = 100000 / C.Np;

if isempty(INITIALIZED) || ~INITIALIZED || Idx == 0 || ~isvalid(Figure) || args.Reset

    Okt23 = datetime(1956,11,23)+[0;1];
    % Okt23 = datetime(2020,11,23)+[0;50];

    Tp = args.Tp;
    if Tp > 0
        while Tp < 14
            Tp = Tp * 2;
        end
        pline_values = R.Date(1:Tp:height(R));
        Pline = @() xline(pline_values,'k','HandleVisibility','off');
    else
        Pline = @() [];
    end

    Plot_Colors
    MyColorMap = [Color_5 ; Color_3 ; Color_2];

    Today = R.Date( min(max(1,Idx),height(R)) );
    Xline = @() xline(Today,'-','','LineWidth',2,'Color',Color_2,'HandleVisibility','off');
    % Cnt(double('A')-1);
    Cnt(0);

    Figure = figure(args.FigNr);
    Figure.Position([3 4]) = args.FigDim;
    Tl = tiledlayout(6,1,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");
        
    Ax = nexttile([2 1]); hold on
    Pl_Imsd = plot(R.Date,R.I*s,'DisplayName','Simulated by PanSim~~~~~');
    if ismember("Iref",R.Properties.VariableNames) && ~all(isnan(R.Iref))
        Pl_Iref = plot(R.Date,R.Iref*s,'LineWidth',1.5,'DisplayName','Prescribed curve');
    end
    if ~all(isnan(Ipred))
        Pl_Iprd = plot(R.Date,Ipred*s,'Color',Color_3,'LineWidth',1.5,'DisplayName','Predicted by MPC');
    else
        Pl_Iprd = plot(R.Date,Ipred*s,'HandleVisibility','off');
    end
    Pl_Irec = plot(R.Date,R.Ir*s,'LineWidth',1.5,'DisplayName','Reconstructed');
    Leg = legend('Location','northoutside','NumColumns',2,'Interpreter','latex','FontSize',12,'Box','off');    
    Xl = [Xl Xline()];
    Pline();
    make_title('Infected')

    Ax = [Ax nexttile([2 1])]; hold on
    if ismember("TrRateBounds",R.Properties.VariableNames)
        Pl_Int = plot_interval(R.Date,R.TrRateBounds(:,1),R.TrRateBounds(:,2));
        Pl_Int(1).HandleVisibility = 'off';
        Pl_Int(2).HandleVisibility = 'off';
        Pl_Int(3).DisplayName = 'Mean range';
    end
    if ismember("TrRateRange",R.Properties.VariableNames)
        Pl_Rnt = plot_interval(R.Date,R.TrRateRange(:,1),R.TrRateRange(:,2),Color_5);
        Pl_Rnt(1).HandleVisibility = 'off';
        Pl_Rnt(2).HandleVisibility = 'off';
        Pl_Rnt(3).DisplayName = '95\% CI';
    end
    Pl_Bmsd = plot(R.Date,R.TrRate,'Color',Color_1,'DisplayName','Measured by PanSim');
    Pl_BMmd = stairs(R.Date,MeanTrRate,'LineWidth',1.5,'Color',Pl_Bmsd.Color*0.8,'DisplayName','... its mean');
    if ismember("TrRateCmd",R.Properties.VariableNames) && ~all(isnan(R.TrRateCmd))
        Pl_Bcmd = stairs(R.Date,R.TrRateCmd,'Color',Color_2,'LineWidth',1.5,'DisplayName','Prescribed');
    end
    % if ismember("TrRateExp",R.Properties.VariableNames) && ~all(isnan(R.TrRateExp))
    %     Pl_Bprd = stairs(R.Date,R.TrRateExp,'Color',Color_3,'DisplayName','Expected');
    % end
    if ~all(isnan(R.TrRateRec))
        Pl_Brec = plot(R.Date,R.TrRateRec,'LineWidth',1.5,'Color',Color_5*0.8,'DisplayName','Reconstructed');
        % Pl_BMrc = stairs(R.Date,MeanTrRateRec,'LineWidth',1.5,'Color',Pl_Brec.Color*0.8,'DisplayName','... its mean');
    else
        % Pl_Bprd = stairs(R.Date,R.TrRateRec,'HandleVisibility','off');
    end
    Leg = legend('Location','northwest','NumColumns',2,'Interpreter','latex','FontSize',12);
    Xl = [Xl Xline()];
    Pline();
    ylim(args.BetaRange)
    make_title('Transmission rate ($\beta$)')

    Ax = [Ax nexttile([2 1])]; hold on
    Ax_NO_GRID = Ax(end);
    YData = 0:width(R.Iq);
    % --
    [DD,YY] = meshgrid(Okt23,YData);
    surf(DD,YY,zeros(width(R.Iq)+1,1) + [0 1])
    % --
    [DD,YY] = meshgrid(R.Date,YData);
    Sf_Iq = surf(DD,YY,R.Iq(:,[1:end,end])');
    Sf_Iq.EdgeAlpha = 0;
    Sf_Iq.FaceAlpha = 0.8;
    Yl = yline(YData,'k');
    colormap(Ax(end),MyColorMap)
    view(Ax(end),[0 90]);
    yticks(YData(2:end)-0.5);
    yticklabels(Vn.policy);
    Xl = [Xl Xline()];
    Xl(end).Color = [0,0,0];
    Pline();
    make_title('Intervention')
    view([0 -90])

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
Pl_Imsd.YData = R.I*s;
assert(numel(Pl_Imsd.XData) == numel(Pl_Imsd.YData))

% Pl_Iprd.XData = R.Date;
Pl_Iprd.YData = Ipred*s;
Pl_Irec.YData = R.Ir*s;

% Pl_Bmsd.XData = R.Date;
Pl_Bmsd.YData = R.TrRate;
assert(numel(Pl_Bmsd.XData) == numel(Pl_Bmsd.YData))

% Pl_Bcmd.XData = Pred.Date;
% Pl_Bcmd.YData = Pred.TrRate;
if ~isempty(Pl_Bcmd)
    Pl_Bcmd.YData = R.TrRateCmd;
end

Pl_BMmd.YDate = MeanTrRate;
% Pl_BMrc.YDate = MeanTrRateRec;

if ~isempty(Pl_Int)
    Pl_Int(1).YData = R.TrRateBounds(:,1);
    Pl_Int(2).YData = R.TrRateBounds(:,2);
    Pl_Int(3).YData = [ Pl_Int(1).YData ; flip(Pl_Int(2).YData) ];
end

if ~isempty(Pl_Rnt)
    Pl_Rnt(1).YData = R.TrRateRange(:,2);
    Pl_Rnt(2).YData = R.TrRateRange(:,1);
    Pl_Rnt(3).YData = [ Pl_Rnt(1).YData ; flip(Pl_Rnt(2).YData) ];
end

% Pl_Bprd.XData = R.Date;
% if ~isempty(Pl_Bprd)
%     Pl_Bprd.YData = R.TrRateExp;
% end
if ~isempty(Pl_Brec)
    Pl_Brec.YData = R.TrRateRec;
end

Sf_Iq.ZData = R.Iq(:,[1:end,end])';

for xl = Xl
    if isvalid(xl)
        xl.Value = R.Date(Idx);
    end
end

fig = Figure;

end

function make_title(str)
    % title(TeX("\makebox[14.5cm]{\textbf{Plot "+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
    title(TeX("\makebox[9.5cm]{\textbf{Plot "+num2str(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
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

