%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 15. (2023a)
%

fp = pcz_mfilename(mfilename("fullpath"));
dirname = fullfile(fp.pdir,"Output","Ctrl_2024-02-27");
dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/" + ...
    "Ctrl_Sum2024-03-09";

DIR_Summary = fullfile(dirname,"Summary");
if ~exist(DIR_Summary,'dir')
    mkdir(DIR_Summary)
end

Results = [
    "C1000_T07"
    "C1000_T14"
    "C500_T01"
    "C500_T07"
    "C500_T14"
    "Erdekes_Teve_T07"
    "Erdekes_Teve_T14"
    "FreeSpread"
    "Ketpupu_Teve_T07"
    "Ketpupu_Teve_T14"
    "Lin1500_T07"
    "Lin1500_T14"
    "Scenario_T07"
    "Scenario_T14"
    "Scenario_T21"
    "Scenario_T28"
    "Sigmoid_T01"
    "Sigmoid_T07"
    "Sigmoid_T14"
    "Sigmoid_T21"
    "Sigmoid_T28"

    % "C500_T01"
    % "C500_T07"
    % "C500_T14"
    % "C1000_T07"
    % "C1000_T14"
    % "Erdekes_Teve_T07"
    % "Erdekes_Teve_T14"
    % "Ketpupu_Teve_T07"
    % "Ketpupu_Teve_T14"
    % "Lin1500_T07"
    % "Lin1500_T14"
    % "Sigmoid_T01"
    % "Sigmoid_T07"
    % "Sigmoid_T14"
    % "Sigmoid_T21"
    % "FreeSpread"
    
    % "Scenario_T07"
    % "Scenario_T14"
    % "Scenario_T21"
    % "Scenario_T28"
    % "Sigmoid_T01"
    % "Sigmoid_T07"
    % "Sigmoid_T14"
    % "Sigmoid_T21"
    % "Sigmoid_T28"
    ]';

for result = Results
%%
    xlsnames = string(cellfun(@(d) {fullfile(d.folder,d.name)}, num2cell(dir(fullfile(dirname,result,"*.xls")))));

    if isempty(xlsnames)
        continue
    end

    n = numel(xlsnames);
    I_cell = cell(1,n);
    b_cell = cell(1,n);

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

        I_cell{i} = R.I;
        b_cell{i} = R.TrRateRec;
    end

    if ismember("Iref",R.Properties.VariableNames)
        Iref = R.Iref;
    else
        Iref = nan(height(R),1);
    end
    Date = R.Date;

    [fig,ret] = Visualize(Date,Iref,I_cell,b_cell);
    
    bname = "_MeanStd_" + result + ".pdf";
    fname = fullfile(dirname,result,bname);
    exportgraphics(fig,fname,'ContentType','vector');

    copyfile(fname,fullfile(DIR_Summary,bname));

end

function opts = detect(xls)
    opts = detectImportOptions(xls);

    SelVarNames = ["Date","TrRate","TrRateRec","Iref","I","Ir"];
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