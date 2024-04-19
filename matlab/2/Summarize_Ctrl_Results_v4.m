%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 15. (2023a)
%

fp = pcz_mfilename(mfilename("fullpath"));
dirname = fullfile(fp.pdir,"Output","Ctrl_2024-02-27");
dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/" + ...
    "Ctrl_Sum2024-04-19";

DIR_Summary = fullfile(dirname,"Summary");
if ~exist(DIR_Summary,'dir')
    mkdir(DIR_Summary)
end


Results = [
    "Scenario2_T21"
    "Scenario3_T21"
    "Scenario4_T21"
    "Scenario1_T30"
    ]';
pattern = '_T(\d+)';

Free1T = readtimetable('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-03-09/FreeSpread/FreeSpread_2024-03-14_08-56.xls');
Free2T = readtimetable('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-04-19/Scenario41_Free_T21/2024-04-19_12-28.xls');
Free2T = Free2T(Free2T.IQ == min(Free2T.IQ),:);

for result = Results
%%
    xlsnames = string(cellfun(@(d) {fullfile(d.folder,d.name)}, num2cell(dir(fullfile(dirname,result,"*.xls")))));

    matches = regexp(result,pattern,'tokens');
    Tp = str2double(matches{1}{1});


    if isempty(xlsnames)
        continue
    end

    n = numel(xlsnames);
    I_cell = cell(1,n);
    b_cell = cell(1,n);

    Iq = [];

    opts = detect(xlsnames(1));
    for i = 1:n
        [~,bname,~] = fileparts(xlsnames(i));
        fprintf('Reading %s/%s\n',result,bname);
        R = readtimetable(xlsnames(i),opts);
        % R = rec_SLPIAHDR(R,'WeightBetaSlope',1e5);

        if any(~ismember(["Ir","TrRateRec"],R.Properties.VariableNames))
            opts = detect(xlsnames(i));
            R = readtimetable(xlsnames(i),opts);
        end

        Iq_k = string(R.Properties.VariableNames(startsWith(R.Properties.VariableNames,'Iq_')));
        R.Iq = R(:,Iq_k).Variables;
        R(:,Iq_k) = [];

        if isempty(Iq)
            Iq = R.Iq;
        else
            Iq = Iq + R.Iq;
        end
        
        I_cell{i} = R.I;
        b_cell{i} = R.TrRateRec;
    end

    Iq = Iq / n;

    if ismember("Iref",R.Properties.VariableNames)
        Iref = R.Iref;
    else
        Iref = nan(height(R),1);
    end
    Date = R.Date;
    Start_Date = Date(1);

    I_Free1 = [ days(Free1T.Date - Start_Date) Free1T.I];
    I_Free2 = [ days(Free2T.Date - Start_Date) Free2T.I];

    if startsWith(result,'Scenario4')
        [fig,ret] = Visualize(Date,Iref,I_cell,b_cell,Iq,"Tp",Tp,"Ifree1",I_Free1,"Ifree2",I_Free2);
    else
        [fig,ret] = Visualize(Date,Iref,I_cell,b_cell,Iq,"Tp",Tp,"Ifree1",I_Free1);
    end
    
    bname = "_MeanStd_" + result;
    fname = fullfile(dirname,result,bname);
    exportgraphics(fig,fname + ".pdf",'ContentType','vector');
    exportgraphics(fig,fname + ".png",'ContentType','image');

    copyfile(fname + ".pdf",fullfile(DIR_Summary,bname + ".pdf"));
    copyfile(fname + ".png",fullfile(DIR_Summary,bname + ".png"));

end

function opts = detect(xls)
    opts = detectImportOptions(xls);

    Iq_k = string(opts.VariableNames(startsWith(opts.VariableNames,'Iq_')));
    SelVarNames = ["Date","TrRate","TrRateRec","Iref","I","Ir",Iq_k];
    for i = numel(SelVarNames):-1:1
        if ~ismember(SelVarNames(i),opts.VariableNames)
            SelVarNames(i) = [];
        end
    end
    opts.SelectedVariableNames = SelVarNames;

    opts = setvartype(opts,opts.SelectedVariableNames,"double");
    opts = setvartype(opts,Vn.policy,"categorical");
    opts = setvartype(opts,"Date","datetime");
    opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");    
end

function [Figure,ret] = Visualize(Date,Iref,I_cell,b_cell,Iq,args)
arguments
    Date
    Iref,I_cell,b_cell
    Iq
    args.Tp = 7
    args.FigDim = [618 752] % [735 566]
    args.FigNr = 16
    args.Reset = true
    args.Ifree1 (:,2) = []
    args.Ifree2 (:,2) = []
end
    Plot_Colors

    Okt23 = datetime(1956,11,23)+[0;1];
    MyColorMap = [Color_5 ; Color_3 ; Color_2];
    MyColorMap = interp1([0;0.5;1],MyColorMap,linspace(0,1,100));

    pline_values = Date(1:args.Tp:numel(Iref));
    Pline = @() xline(pline_values,'k','HandleVisibility','off');
    
    Cnt(double('A')-1);

    I = [I_cell{:}];
    I_std = std(I,0,2);
    I_mean = mean(I,2);

    beta = [b_cell{:}];
    beta_std = std(beta,0,2);
    beta_mean = mean(beta,2);

    Figure = figure(args.FigNr);
    Figure.Position([3 4]) = args.FigDim;
    Tl = tiledlayout(4,1,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");
        
    Ax = nexttile([2 1]); hold on
    plot(Okt23,[1,1],'.w','DisplayName','Intervention')
    fill(Okt23,[1,1],Color_5,'DisplayName','~low')
    fill(Okt23,[1,1],Color_3,'DisplayName','~medium')
    fill(Okt23,[1,1],Color_2,'DisplayName','~high')
    plot(Okt23,[1,1],'.w','DisplayName','Number of infected ($\mathbf{I}_k$)')
    Pl_Imsd = plot(Date,I,'Color',[1,1,1]*0.75,'HandleVisibility','off','DisplayName','~simulations Pansim');
    Pl_Imsd(1).HandleVisibility = 'on';
    Pl_Int = plot_mean_var(Date,I_mean,I_std,Color_3,"Alpha",2,"LineWidth",2);
    Pl_Int(1).HandleVisibility = 'on';
    Pl_Int(1).DisplayName = '~mean curve';
    Pl_Int(2).HandleVisibility = 'off';
    Pl_Int(3).HandleVisibility = 'off';
    Pl_Int(4).DisplayName = '~$\pm$2\,std (95\%CI)';
    plot(Okt23,[1,1],'.w','DisplayName','')
    Pl_Iref = plot(Date,Iref,'LineWidth',2,'Color',Color_2,'DisplayName','Prescribed curve~~~~~~~~~~~~~~~~');

    if ~isempty(args.Ifree1)
        Pl_Ifree1 = plot(args.Ifree1(:,1) + Date(1),args.Ifree1(:,2),'Color',Color_5,'LineWidth',1.5, ...
            'DisplayName','Free spread');
    end
    
    if ~isempty(args.Ifree2)
        Pl_Ifree2 = plot(args.Ifree2(:,1) + Date(1),args.Ifree2(:,2),'Color',Color_6,'LineWidth',1.5, ...
            'DisplayName','Free spread of 2nd wave');
    end
 
    ILim = [0,3500];
    ylim(ILim)
    
    make_title('Infected')
    LegI = legend('Location','northoutside','Interpreter','latex','FontSize',12,'NumColumns',3,'Box','off');    


    Ax_NO_GRID = nexttile; hold on
    YData = 0:width(Iq);
    % --
    [DD,YY] = meshgrid(Okt23,YData);
    surf(DD,YY,zeros(width(Iq)+1,1) + [0 1],'HandleVisibility','off')
    % --
    [DD,YY] = meshgrid(Date,YData);
    Sf_Iq = surf(DD,YY,Iq(:,[1:end,end])','HandleVisibility','off');
    Sf_Iq.EdgeAlpha = 0;
    Sf_Iq.FaceAlpha = 0.8;
    Yl = yline(YData,'k','HandleVisibility','off');
    colormap(Ax_NO_GRID,MyColorMap)
    view(Ax_NO_GRID,[0 90]);
    yticks(YData(2:end)-0.5);
    yticklabels(Vn.policy);
    make_title('Average strictness of interventions')
    view([0 -90])
    Pline()

    XLim = [Date([1,end])];
    xlim(XLim)

    
    Ax = [Ax nexttile]; hold on
    Pl_Bmsd = plot(Date,beta,'Color',[1,1,1]*0.75,'HandleVisibility','off','DisplayName','Reconstructed');
    Pl_Bmsd(1).HandleVisibility = 'on';
    Pl_Bnt = plot_mean_var(Date,beta_mean,beta_std,Color_3,"Alpha",2,"LineWidth",2);
    Pl_Bnt(1).HandleVisibility = 'on';
    Pl_Bnt(1).DisplayName = 'Mean curve';
    Pl_Bnt(2).HandleVisibility = 'off';
    Pl_Bnt(2).DisplayName = 'Mean curve';
    Pl_Bnt(3).HandleVisibility = 'off';
    Pl_Bnt(3).DisplayName = 'Mean curve';
    Pl_Bnt(4).DisplayName = '$\pm$2\,std (95\%CI)';
    make_title('Transmission rate')
    % LegB = legend('Location','northwest','Interpreter','latex','FontSize',12,'NumColumns',3,'Box','off');    

    BLim = [0,0.4];
    ylim(BLim)

    ret.Link_XLim = linkprop(Ax,'XLim');
    
    
    for ax = Ax
        grid(ax,'on')
        box(ax,'on')
    end
    
    for ax = [ Ax Ax_NO_GRID ]
        ax.TickLabelInterpreter = "latex";
        ax.FontSize = 12;
        
        if ax ~= Ax_NO_GRID
            ax.XMinorGrid = 'on';
        end
        ax.XTick = Date(day(Date) == 1 & mod(month(Date),2)==0);
        ax.XAxis.MinorTick = 'off';
        ax.XAxis.MinorTickValues = Date(weekday(Date) == 1); 
        ax.XLim = XLim;
    end


end

function make_title(str)
    title(TeX("\makebox[10.5cm]{\textbf{"+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end
