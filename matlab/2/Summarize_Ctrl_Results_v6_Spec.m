%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 15. (2023a)
%
% Ez az abra kerul bele a cikkbe
% 
%  !!!!!!!!!!!! VEDETT !!!!!!!!!!

if false
%%
    fp = pcz_mfilename(mfilename("fullpath"));
    dirname = fullfile(fp.pdir,"Output","Ctrl_2024-02-27");
    dirname = "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/" + ...
        "Ctrl_Sum2024-04-19";
    
    DIR_Summary = fullfile(dirname,"Summary");
    if ~exist(DIR_Summary,'dir')
        mkdir(DIR_Summary)
    end
    
    
    Results = [
        "Scenario1_T30"
        "Scenario3_T21"
        "Scenario2_T21"
        "Scenario4_T21"
        ]';
    pattern = '_T(\d+)';
    
    T_fsp = load_free_spread(...
        "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-03-09/FreeSpread");
    
    T_fx1 = load_free_spread(...
        "/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Ctrl_Sum2024-04-19/Scenario41_Fixed_T21");
    
    T_fx1 = T_fx1(T_fx1.IQ == min(T_fx1.IQ),:);
    
    R = cell(1,4);
    Tp = zeros(1,4);
    for i = 1:4
        R{i} = load_ctrled_spread(fullfile(dirname,Results(i)));
        matches = regexp(Results(i),pattern,'tokens');
        Tp(i) = str2double(matches{1}{1});
    end

    N = height(R{2})-1;
    t_sim = 0:N;
        
    Mean = 123;
    Std = 34;
    Peak = 2200;
    Iref = normpdf(t_sim,Mean,Std)';
    Iref = Iref / max(Iref) * Peak;

    R{2}.Iref = Iref;
end

%%

[fig,links] = Visualize(R,"Tp",Tp,"Tcmp1",T_fsp,"Tcmp2",T_fx1);
% exportgraphics(fig,fullfile(DIR_Summary,"Summary.pdf"));
% exportgraphics(fig,fullfile(DIR_Summary,"Summary.png"));


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

function R = load_ctrled_spread(dirname)

    xlsnames = dir(fullfile(dirname,"*.xls"));
    xlsnames = cellfun(@(s) {fullfile(s.folder,s.name)},num2cell(xlsnames));

    n = numel(xlsnames);
    opts = detect(xlsnames{1});
    R = readtimetable(xlsnames{1},opts);
    N = height(R);

    I = zeros(N,n);
    b = zeros(N,n);
    Iq = zeros(N,numel(Vn.policy),n);

    for i = 1:n
        [~,bname,~] = fileparts(xlsnames(i));
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
    end

    R = R(:,"Iref");
    % --
    R.I = I;
    R.mean_I = mean(I,2);
    R.std_I = std(I,[],2);
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

function [Figure,ret] = Visualize(Rs,args)
arguments
    Rs
    args.Tp = 7
    args.FigDim = [1070 1300] % [735 566]
    args.FigNr = 16
    args.Reset = true
    args.Tcmp1 = []
    args.Tcmp2 = []
    args.LabelOffset = 0;
end
    Okt23 = datetime(1956,11,23)+[0;1];
    MyColorMap = [Col.r_5 ; Col.r_3 ; Col.r_2];
    MyColorMap = interp1([0;0.5;1],MyColorMap,linspace(0,1,100));
    
    height_A = 13;
    height_B = 10;
    height_C = 10;
    height_D = 9;

    height = height_A + height_B + height_C + height_D;
    height = 2*height + 1;

    Figure = figure(args.FigNr);
    Figure.Position([3 4]) = args.FigDim;
    Tl = tiledlayout(height,2,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");
        
    Ax = nexttile([height_A 1]); hold on

    %% Legend

    LegEntry = @(S) plot(Okt23,[1,1],'.w','DisplayName',S);

    LegEntry('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    LegEntry('')
    LegEntry('')
    LegEntry('')

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
    patch(Okt23([1 2 2 1]),[1 1 2 2],Col.r_6, ...
            'EdgeColor',Col.r_6, ...
            'FaceAlpha',0.2, ...
            'EdgeAlpha',1, ...
            'LineWidth',1.5, ...
            'DisplayName','~Free spread of 2nd wave in Plot \textbf{D1} (mean\,$\pm$\,2\,std)~~');


    plot(Okt23,[1,1],'Color',Color.Light_Gray,'DisplayName','~Multiple simulations in PanSim~~');
    plot(Okt23,[1,1],'Color',Color.Black,'DisplayName','~Highlighted curves of controlled');
    LegEntry('~simulations resulted by interventions')
    LegEntry('~in Plots \textbf{\{A,B,C,D\}2}')
    
    plot(Okt23,[1,1],'.w','DisplayName','Intervention')
    fill(Okt23,[1,1],Col.r_5,'DisplayName','~Low')
    fill(Okt23,[1,1],Col.r_3,'DisplayName','~Medium')
    fill(Okt23,[1,1],Col.r_2,'DisplayName','~High')
    
    LegI = legend('Location','northoutside','Interpreter','latex','FontSize',12,'NumColumns',4,'Box','off');    


    if false
        %%
        Visualize(R,"Tp",Tp,"Tcmp1",T_fsp,"Tcmp2",T_fx1);
    end
    
    Names = [
        "Flatten the curve"
        "Almost free but delayed"
        "We give more time"
        "Two waves"
        ];

    LabelOffset = [0 2 1 3];
    for i = 1:numel(Rs)
        R = Rs{i};
    
        Label = @(j) [ char('A' + LabelOffset(i)) num2str(j) ];
        Cnt(double('A')-1 + 4*LabelOffset(i));

        Date = R.Date;
        pline_values = Date(1:args.Tp(i):numel(R.Iref));
        Pline = @() xline(pline_values,'k','HandleVisibility','off');
    
        %% Plot A.
    
        if i > 1
            Ax = [Ax nexttile([height_A,1])]; hold on
        end

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
     
        ILim = [0,3500];
        ylim(ILim)
        
        make_title_labeled(Label(1),"Infected \textit{(" + Names(i) + ")}");
    
        %% Plot B.
    
        Ax_NO_GRID = nexttile([height_B,1]); hold on
        YData = 0:width(R.mean_Iq);
        % --
        [DD,YY] = meshgrid(Okt23,YData);
        surf(DD,YY,zeros(width(R.typical_Iq)+1,1) + [0 1],'HandleVisibility','off')
        % --
        [DD,YY] = meshgrid(Date,YData);
        Sf_Iq = surf(DD,YY,R.typical_Iq(:,[1:end,end])','HandleVisibility','off');
        Sf_Iq.EdgeAlpha = 0;
        Sf_Iq.FaceAlpha = 0.8;
        Yl = yline(YData,'k','HandleVisibility','off');
        colormap(Ax_NO_GRID,MyColorMap)
        view(Ax_NO_GRID,[0 90]);
        yticks(YData(2:end)-0.5);
        yticklabels(Vn.policy);
        make_title_labeled(Label(2),'Strictness of a selected (black) curve')
        view([0 -90])
        Pline()
    
        XLim = [Date([1,end])];
        xlim(XLim)
    
        %% Plot C.
    
        Ax_NO_GRID = [Ax_NO_GRID nexttile([height_C,1])]; hold on
        YData = 0:width(R.mean_Iq);
        % --
        [DD,YY] = meshgrid(Okt23,YData);
        surf(DD,YY,zeros(width(R.mean_Iq)+1,1) + [0 1],'HandleVisibility','off')
        % --
        [DD,YY] = meshgrid(Date,YData);
        Sf_Iq = surf(DD,YY,R.mean_Iq(:,[1:end,end])','HandleVisibility','off');
        Sf_Iq.EdgeAlpha = 0;
        Sf_Iq.FaceAlpha = 0.8;
        Yl = yline(YData,'k','HandleVisibility','off');
        colormap(Ax_NO_GRID(end),MyColorMap)
        view(Ax_NO_GRID(end),[0 90]);
        yticks(YData(2:end)-0.5);
        yticklabels(Vn.policy);
        make_title_labeled(Label(3),'Averaged strictness of all interventions')
        % make_title('Average strictness of interventions ($\Rightarrow$ controlled curves)')
        view([0 -90])
        Pline()
    
        XLim = [Date([1,end])];
        xlim(XLim)
    
        %% Plot D.
    
        Ax = [Ax nexttile([height_D,1])]; hold on
        Pl_Bmsd = plot(R.Date,R.beta,'Color',[1,1,1]*0.75,'HandleVisibility','off','DisplayName','Reconstructed');
        Pl_Bmsd(1).HandleVisibility = 'on';
        Pl_Bnt = plot_mean_var(R.Date,R.mean_beta,R.std_beta,Col.r_3,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
        Pl_Bnt(1).HandleVisibility = 'on';
        Pl_Bnt(1).DisplayName = 'Mean curve';
        Pl_Bnt(2).HandleVisibility = 'off';
        Pl_Bnt(2).DisplayName = 'Mean curve';
        Pl_Bnt(3).HandleVisibility = 'off';
        Pl_Bnt(3).DisplayName = 'Mean curve';
        Pl_Bnt(4).DisplayName = '$\pm$2\,std (95\%CI)';
        make_title_labeled(Label(4),'Approximated transmission rate')
        Ax(end).YTick = 0:0.1:0.5;
        % LegB = legend('Location','northwest','Interpreter','latex','FontSize',12,'NumColumns',3,'Box','off');    
    
        BLim = [0,0.3];
        ylim(BLim)
    
        if i == 1
            bx = nexttile([1,2]);
            hold on
            plot([0,1],[0,0]+30,'Color',Color.Dark_Gray);
            bx.YLim = [0,1];
            bx.XLim = [0,1];
            bx.Clipping = "off";
            bx.Visible = "off";
        end

        %% Transform axes
    
        ret.Link_XLim = linkprop(Ax,'XLim');
        
        for ax = Ax
            grid(ax,'on')
            box(ax,'on')
        end
        
        for ax = [ Ax Ax_NO_GRID ]
            ax.TickLabelInterpreter = "latex";
            ax.FontSize = 12;
            
            if ~ismember(ax,Ax_NO_GRID)
                ax.XMinorGrid = 'on';
            else
                ax.YAxis.FontSize = 10;
            end
            ax.XTick = Date(day(Date) == 1 & mod(month(Date),2)==0);
            ax.XAxis.MinorTick = 'off';
            ax.XAxis.MinorTickValues = Date(weekday(Date) == 1); 
            ax.XLim = XLim;
        end
    end

end

function make_title(str)
    title(TeX("\makebox[8cm]{\textbf{"+char(Cnt)+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end

function make_title_labeled(label,str)
    title(TeX("\makebox[8cm]{\textbf{"+label+".} " + str + "\hfill}"),"FontSize",13,"Interpreter","latex")
end
