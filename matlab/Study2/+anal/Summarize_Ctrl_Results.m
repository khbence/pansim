%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 15. (2023a)
%

DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2";

DIR_Summary = fullfile(DIR,"Summary");
if ~exist(DIR_Summary,'dir')
    mkdir(DIR_Summary)
end

Results = [
    "C590"
    "C1090"
    "Ketpupu_Teve"
    "Lin1500"
    "Sigmoid900"
    ]';

for result = Results
%%
    xlsnames = string(cellfun(@(d) {fullfile(d.folder,d.name)}, num2cell(dir(fullfile(DIR,result,"*.xls")))));

    n = numel(xlsnames);
    I_cell = cell(1,n);
    b_cell = cell(1,n);

    opts = detect(xlsnames(1));
    for i = 1:n
        R = readtimetable(xlsnames(i),opts);
        I_cell{i} = R.Ir;
        b_cell{i} = R.TrRateRec;
    end

    Iref = R.Iref;
    Date = R.Date;

    [fig,ret] = Visualize(Date,Iref,I_cell,b_cell);
    
    bname = "MeanStd_" + result + ".pdf";
    fname = fullfile(DIR,result,bname);
    exportgraphics(fig,fname,'ContentType','vector');

    copyfile(fname,fullfile(DIR_Summary,bname));

end

function opts = detect(xls)
    opts = detectImportOptions(xls);
    opts = setvartype(opts,opts.SelectedVariableNames,"double");
    opts = setvartype(opts,Vn.policy,"categorical");
    opts = setvartype(opts,"Date","datetime");
    opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
    opts.SelectedVariableNames = ["Date","TrRate","TrRateRec","Iref","I","Ir"];
end

function [Figure,ret] = Visualize(Date,Iref,I_cell,b_cell,args)
arguments
    Date
    Iref,I_cell,b_cell
    args.Tp = 7
    args.FigDim = [735 566]
    args.FigNr = 16
    args.Reset = true
end
    Plot_Colors

    Cnt(double('A')-1);

    I = [I_cell{:}];
    I_std = std(I,0,2);
    I_mean = mean(I,2);

    beta = [b_cell{:}];
    beta_std = std(beta,0,2);
    beta_mean = mean(beta,2);

    Figure = figure(args.FigNr);
    Figure.Position([3 4]) = args.FigDim;
    Tl = tiledlayout(3,1,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");
        
    Ax = nexttile([2 1]); hold on
    Pl_Imsd = plot(Date,I,'Color',[1,1,1]*0.75,'HandleVisibility','off','DisplayName','Simulated by Pansim');
    Pl_Imsd(1).HandleVisibility = 'on';
    Pl_Int = plot_mean_var(Date,I_mean,I_std,Color_3,"Alpha",2,"LineWidth",2);
    Pl_Int(1).HandleVisibility = 'on';
    Pl_Int(1).DisplayName = 'Mean curve';
    Pl_Int(2).HandleVisibility = 'off';
    Pl_Int(2).DisplayName = 'Mean curve';
    Pl_Int(3).HandleVisibility = 'off';
    Pl_Int(3).DisplayName = 'Mean curve';
    Pl_Int(4).DisplayName = '95\% CI';
    Pl_Iref = plot(Date,Iref,'LineWidth',2,'Color',Color_2,'DisplayName','Prescribed curve');
    make_title('Infected')
    LegI = legend('Location','northwest','Interpreter','latex','FontSize',12,'NumColumns',4,'Box','off');    
        
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
    Pl_Bnt(4).DisplayName = '95\% CI';
    make_title('Transmission rate')
    LegB = legend('Location','northwest','Interpreter','latex','FontSize',12,'NumColumns',3,'Box','off');    

    ret.Link_XLim = linkprop(Ax,'XLim');
    
    XLim = [Date([1,end])];
    
    for ax = Ax
        grid(ax,'on')
        box(ax,'on')
    end
    
    for ax = Ax
        ax.TickLabelInterpreter = "latex";
        ax.FontSize = 12;
        
        ax.XTick = Date(day(Date) == 1 & mod(month(Date),2)==0);
        ax.XMinorGrid = 'on';
        ax.XAxis.MinorTick = 'off';
        ax.XAxis.MinorTickValues = Date(weekday(Date) == 1); 
        ax.XLim = XLim;
    end


end

function make_title(str)
    title(TeX("\makebox[10.5cm]{\textbf{"+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end
