%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 15. (2023a)
%

if false
%%
    fp = pcz_mfilename(mfilename("fullpath"));
    dirname = fullfile(fp.pdir,"Output","Ctrl_2024-02-27");
    dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/" + ...
        "Ctrl_Sum2024-05-30";
    
    DIR_Summary = fullfile(dirname,"Summary");
    if ~exist(DIR_Summary,'dir')
        mkdir(DIR_Summary)
    end
    
    Results = [
        % "Simple_Scenario4_T14"
        % "NewScenario1_Scenario4_T14"
        % "ContWithOmicron_Scenario4_T14"
        % "ContWithOmicron_noMtp_Scenario4_T14"
        "CtrlOmicron55_Flatten_T07"
        "CtrlOmicron55_Flatten_T14"
        "CtrlOmicron55_Flatten_T21"
        "CtrlAlpha55_Flatten_T07"
        "CtrlAlpha55_Flatten_T14"
        "CtrlAlpha55_Flatten_T21"
        "CtrlAlpha55_Flatten_T30"
        ]';
    pattern = '_T(\d+)';
    
    T_fsp_omicron = load_free_spread(...
        "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-05-30/FreeOmicron55");
    
    T_fsp_alpha = load_free_spread(...
        "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-05-30/FreeAlpha55_210");
    
    % T_fx1 = load_free_spread(...
    %     "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-04-19/Scenario41_Fixed_T21");
    % 
    % T_fx1 = T_fx1(T_fx1.IQ == min(T_fx1.IQ),:);
    
    R = cell(1,numel(Results));
    Tp = zeros(1,numel(Results));
    for i = 1:numel(Results)
        matches = regexp(Results(i),pattern,'tokens');
        Tp(i) = str2double(matches{1}{1});
        R{i} = load_ctrled_spread(fullfile(dirname,Results(i)),Tp(i));
    end

end

%%

Names = {
    ""  6000 "Omicron emerges, weekly interventions"
    ""  6000 "Omicron emerges, new interventions every 2nd week"
    ""  6000 "Omicron emerges, new interventions every 3rd week"
    ""  6000 "Alpha emerges, weekly interventions"
    ""  6000 "Alpha emerges, new interventions every 2nd week"
    ""  6000 "Alpha emerges, new interventions every 3rd week"
    ""  6000 "Alpha emerges, monthly interventions"
    };

[fig] = Visualize(R,Names,1:3,"Tp",Tp,"Tcmp1",T_fsp_omicron, ... "Tcmp2",T_fx1,
    "FigNr",123,"FigDim",[500 700]);
exportgraphics(fig,fullfile(DIR_Summary,"Omicron55.pdf"));
exportgraphics(fig,fullfile(DIR_Summary,"Omicron55.png"));

[fig] = Visualize(R,Names,4:7,"Tp",Tp,"Tcmp1",T_fsp_alpha, ... "Tcmp2",T_fx1,
    "FigNr",124,"FigDim",[500 870]);
exportgraphics(fig,fullfile(DIR_Summary,"Alpha55.pdf"));
exportgraphics(fig,fullfile(DIR_Summary,"Alpha55.png"));

% [fig,links] = Visualize(R,Names,[2,4],"Tp",Tp,"Tcmp1",T_fsp, ... "Tcmp2",T_fx1,
%     "FigNr",124);
% exportgraphics(fig,fullfile(DIR_Summary,"Summary24.pdf"));
% exportgraphics(fig,fullfile(DIR_Summary,"Summary24.png"));


function opts = detect(xls)
    opts = detectImportOptions(xls);

    Iq_k = string(opts.VariableNames(startsWith(opts.VariableNames,'Iq_')));
    SelVarNames = ["Date","TrRate","TrRateRec","Iref","I","Ir","TrRateBounds_2",Iq_k];
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

function R = load_ctrled_spread(dirname,Tp)

    xlsnames = dir(fullfile(dirname,"*.xls"));
    xlsnames = cellfun(@(s) {fullfile(s.folder,s.name)},num2cell(xlsnames));

    n = numel(xlsnames);
    opts = detect(xlsnames{1});
    R = readtimetable(xlsnames{1},opts);
    N = height(R);

    I = zeros(N,n);
    b = zeros(N,n);
    Iq = zeros(N,numel(Vn.policy),n);
    IncMtp = zeros(N,1);

    for i = 1:n
        [~,bname,~] = fileparts(xlsnames{i});
        [~,result,~] = fileparts(dirname);
        fprintf('Reading %s/%s\n',result,bname);
        R = readtimetable(xlsnames{i},opts);

        if any(~ismember(["Ir","TrRateRec"],R.Properties.VariableNames))
            opts = detect(xlsnames{i});
            R = readtimetable(xlsnames{i},opts);
        end

        Iq_k = string(R.Properties.VariableNames(startsWith(R.Properties.VariableNames,'Iq_')));
        Iq(:,:,i) = R(:,Iq_k).Variables;
        
        I(:,i) = R.I;
        b(:,i) = R.TrRateRec;

        % An increase in the beta_multiplier detected
        Idx = find(R.TrRateBounds_2(2:end) ./ R.TrRateBounds_2(1:end-1) > 1) + 1;
        IncMtp(Idx) = IncMtp(Idx) + 1;
    end

    IncMtp = IncMtp ./ n;
    IncMtp(IncMtp < 0.2) = 0;

    R = R(:,"Iref");
    % --
    R.I = I;
    R.mean_I = mean(I,2);
    R.std_I = std(I,[],2);
    R.IncMtp = IncMtp;
    % --
    [~,idx1] = min(vecnorm(I - R.Iref,2,1));
    R.closest_I = I(:,idx1);
    [~,idx2] = min(vecnorm(I - R.mean_I,2,1));
    R.typical_I = I(:,idx2);
    % --
    R.beta = b;
    R.mean_beta = mean(b,2);
    R.std_beta = std(b,[],2);
    % --
    R.mean_Iq = mean(Iq,3);
    R.mode_Iq = mode(Iq,3);
    R.median_Iq = median(Iq,3);
    % --
    R.closest_Iq = Iq(:,:,idx1);
    R.typical_Iq = Iq(:,:,idx2);

end

function R = load_free_spread(dirname)

    xlsnames = dir(fullfile(dirname,"*.xls"));
    xlsnames = cellfun(@(s) {fullfile(s.folder,s.name)},num2cell(xlsnames));
    
    opts = detectImportOptions(xlsnames{1});
    opts.SelectedVariableNames = ["Date","I","IQ"];
    opts = setvartype(opts,"IQ","int32");
    opts = setvartype(opts,"I","double");
    opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
    
    R = readtimetable(xlsnames{1},opts);
    n = numel(xlsnames);
    
    I = zeros(height(R),n);
    I(:,1) = R.I;
    
    for i = 2:n
        R = readtimetable(xlsnames{i},opts);
        I(:,i) = R.I;
    end
    
    R.I = I;
    R.Mean = mean(I,2);
    R.Std = std(I,0,2);

end

function [Figure] = Visualize(Rs,Names,Idx,args)
arguments
    Rs
    Names
    Idx
    args.Tp = 7
    args.FigDim = [500 700] % [735 566]
    args.FigNr = 16
    args.Reset = true
    args.Tcmp1 = []
    args.Tcmp2 = []
    args.LabelOffset = 0;
end
    Okt23 = [-2,-1];
    MyColorMap = [Col.r_5 ; Col.r_3 ; Col.r_2];
    MyColorMap = interp1([0;0.5;1],MyColorMap,linspace(0,1,100));
    
    height_Top = 4;
    height_I = 20;
    height_Iq = 4;
    height_Pd1 = 7;
    height_Pd = 10;

    height_all = numel(Idx)*(height_Iq + height_I + height_Pd) - height_Pd + height_Pd1 + height_Top;

    Figure = figure(args.FigNr);
    Figure.Position([3 4]) = args.FigDim;
    Tl = tiledlayout(height_all,1,"TileSpacing","none","Padding","compact","TileIndexing","columnmajor");
        
    ax = nexttile([height_Top,1]);
    hold on, grid on, box on
    ax.Visible = 'off';

    %% Legend

    LegEntry = @(S) plot(Okt23,[1,1],'.w','DisplayName',S);

    % LegEntry('Simple curves:')
    plot(Okt23,[1,1],'LineWidth',2,'Color',Col.r_2,'DisplayName','~Prescribed curve ($\mathbf{I}_k^{\mathrm{Ref}}$)~~');

    % LegEntry('Mean\,$\pm$\,2\,std curves:')
    patch(Okt23([1 2 2 1]),[1 1 2 2],Col.r_3, ...
            'EdgeColor',Col.r_3, ...
            'FaceAlpha',0.2, ...
            'EdgeAlpha',1, ...
            'LineWidth',1.5, ...
            'DisplayName','~Controlled spread through interventions (mean\,$\pm$\,2\,std)~');
    patch(Okt23([1 2 2 1]),[1 1 2 2],Col.r_5, ...
            'EdgeColor',Col.r_5, ...
            'FaceAlpha',0.2, ...
            'EdgeAlpha',1, ...
            'LineWidth',1.5, ...
            'DisplayName','~Free spread (mean\,$\pm$\,2\,std)');

    plot(Okt23,[1,1],'Color',Color.Light_Gray,'DisplayName','~Multiple simulations in PanSim~~');

    xline(Okt23(1),'Color',COL.Color_Brown,'DisplayName','~(vertical) Increased transmission rate detected');

    plot(Okt23,[1,1],'.w','DisplayName','Intervention','HandleVisibility','off')
    fill([0 1 1 0]+0,[1,1,2,2],Col.r_5,'DisplayName','~Low','HandleVisibility','off')
    fill([0 1 1 0]+2.6,[1,1,2,2],Col.r_3,'DisplayName','~Medium','HandleVisibility','off')
    fill([0 1 1 0]+6.1,[1,1,2,2],Col.r_2,'DisplayName','~High','HandleVisibility','off')
    text(-0.2,2.8,'Overall strictness of interventions:','Interpreter','latex','FontSize',12)
    text(1.1,1.4,'Low (0)','Interpreter','latex','FontSize',12)
    text(3.7,1.4,'Medium (0.5)','Interpreter','latex','FontSize',12)
    text(7.2,1.4,'High (1)','Interpreter','latex','FontSize',12)

    LegI = legend('Location','northoutside','Interpreter','latex','FontSize',12,'NumColumns',1,'Box','off');    
    

    Date = Rs{1}.Date;

    XLim = [Date([1,end])];
    ax.XLim = [0,9];
    ax.YLim = [1,2.5];

    
    for i = Idx
        R = Rs{i};
    
        Label = @(j) Names{i,1} + num2str(j);
        % Cnt(double('A')-1 + 4*LabelOffset(i));

        pline_values = Date(1:args.Tp(i):height(Rs{i}));
        Pline = @() xline(pline_values,'k','HandleVisibility','off');

        %%

        if i == Idx(1)
            ax = nexttile([height_Pd1,1]);
            ax.Visible = 'off';
        else
            ax = nexttile([height_Pd,1]);
            ax.Visible = 'off';
        end

        %%

        ax = nexttile([height_Iq,1]);
        hold on, grid on, box on

        hold on, grid on, box on
        
        mean_Iq = mean(R.mean_Iq,2);
        CData = [mean_Iq, mean_Iq];

        YData = 0:1;
        % --
        [DD,YY] = meshgrid(Okt23,YData);
        surf(DD,YY,zeros(2,1) + [0 1],'HandleVisibility','off')
        % --
        [DD,YY] = meshgrid(Date,YData);
        Sf_Iq = surf(DD,YY,CData','HandleVisibility','off');
        Sf_Iq.EdgeAlpha = 0;
        Sf_Iq.FaceAlpha = 1;
        Yl = yline(YData,'k','HandleVisibility','off');
        colormap(ax,MyColorMap)
        % yticks(YData(2:end)-0.5);
        % yticklabels(Vn.policy);
        ax.XTickLabels = {};
        ax.YTickLabels = {};
        Pline()        

        make_title(Label(1),Names{i,3});
        
        ax.TickLabelInterpreter = "latex";
        ax.FontSize = 12;        
        ax.YAxis.FontSize = 10;
        ax.XTick = Date(day(Date) == 1 & mod(month(Date),2)==0);
        ax.XAxis.MinorTick = 'off';
        ax.XAxis.MinorTickValues = Date(weekday(Date) == 1); 
        ax.XLim = XLim;
        
        %% Plot I
    
        ax = nexttile([height_I,1]);
        hold on; grid on; box on
 
        Idx = find(R.IncMtp > 0.23);
        Idx(Idx < 55) = [];
        xline(R.Date(Idx),'Color',COL.Color_Brown);
        
        if ~isempty(args.Tcmp1)
            plot(args.Tcmp1.Date,args.Tcmp1.I,'Color',Color.Light_Gray,'HandleVisibility','off');
            Hide = plot_mean_var(args.Tcmp1.Date,args.Tcmp1.Mean,args.Tcmp1.Std,Col.r_5,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
            for pl = Hide'
                pl.HandleVisibility = 'off';
            end
        end
        
        if ~isempty(args.Tcmp2) && i == 4
            plot(args.Tcmp2.Date,args.Tcmp2.I,'Color',Color.Light_Gray,'HandleVisibility','off');
            Hide = plot_mean_var(args.Tcmp2.Date,args.Tcmp2.Mean,args.Tcmp2.Std,Col.r_6,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
            for pl = Hide'
                pl.HandleVisibility = 'off';
            end
        end
        
        plot(R.Date,R.I,'Color',Color.Light_Gray,'HandleVisibility','off');
    
        Hide = plot_mean_var(R.Date,R.mean_I,R.std_I,Col.r_3,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
        for pl = Hide'
            pl.HandleVisibility = 'off';
        end
        plot(R.Date,R.typical_I,'Color',Color.Black,'HandleVisibility','off');
    
        plot(Date,R.Iref,'LineWidth',2,'Color',Col.r_2,'HandleVisibility','off');
     
        ILim = [0,Names{i,2}];
        ylim(ILim)
    
        ax.TickLabelInterpreter = "latex";
        ax.FontSize = 12;
        
        ax.XMinorGrid = 'on';
        ax.XTick = Date(day(Date) == 1 & mod(month(Date),2)==0);
        ax.XAxis.MinorTick = 'off';
        ax.XAxis.MinorTickValues = Date(weekday(Date) == 1); 
        ax.XLim = XLim;

    end

end

function make_title(~,str)
    title(TeX("\makebox[8cm]{" + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end

function make_title_labeled(label,str)
    title(TeX("\makebox[8cm]{\textbf{"+label+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end
