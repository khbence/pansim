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
        "FreeAlpha70_210_7days"
        "Alpha70_7days__Sc1typical_T30"
        "CtrlAlpha70_7days_Flatten_T07"
        "CtrlAlpha70_7days_Flatten_T14"
        "CtrlAlpha70_7days_Flatten_T21"
        "CtrlAlpha70_7days_Flatten_T30"

        "FreeOmicron70_210_7days"
        "Omicron70_7days__Sc1typical_T30"
        "CtrlOmicron70_7days_Flatten_T07"
        "CtrlOmicron70_7days_Flatten_T14"
        "CtrlOmicron70_7days_Flatten_T21"
        "CtrlOmicron70_7days_Flatten_T30"

        "FreeStartOmicron_60"
        "StartOmicron__Sc1typical_T30"
        "CtrlStartOmicron_Flatten_T07"
        "CtrlStartOmicron_Flatten_T14"
        "CtrlStartOmicron_Flatten_T21"
        "CtrlStartOmicron_Flatten_T30"
        ]';

    FreeSpreads = [
        ]';

    pattern = '_T(\d+)';
    
    % Load controlled spread records
    R = cell(1,numel(Results));
    Tp = zeros(1,numel(Results));
    for i = 1:numel(Results)
        matches = regexp(Results(i),pattern,'tokens');
        try
            Tp(i) = str2double(matches{1}{1});
        catch
            Tp(i) = 14;
        end
        R{i} = load_ctrled_spread(fullfile(dirname,Results(i)),Tp(i));
    end

    % Load free spread records
    % F = cell(1,numel(FreeSpreads));
    % for i = 1:numel(FreeSpreads)
    %     F{i} = load_ctrled_spread(fullfile(dirname,FreeSpreads(i)),14);
    % end

    % %%

    T = table();

    for i = 1:numel(R)
        s = R{i}.Properties.UserData;
        [~,s.DirName,~] = fileparts(s.DirName);
        s = rmfield(s,["Typical","Closest","TypicalIdx","ClosestIdx"]);
        Ts = struct2table(s,AsArray=true);
    
        if isempty(T)
            T = Ts;
            continue
        end

        Vnt = T.Properties.VariableNames;
        Vns = Ts.Properties.VariableNames;
    
        NewVars = string(setdiff(Vns,Vnt));
        NaNs = repmat({nan(height(T),1)},[1,numel(NewVars)]);
        T = addvars(T,NaNs{:},'NewVariableNames',NewVars);

        NewVars = string(setdiff(Vnt,Vns));
        NaNs = repmat({nan(height(Ts),1)},[1,numel(NewVars)]);
        Ts = addvars(Ts,NaNs{:},'NewVariableNames',NewVars);

        T = [T ; Ts];
    end
    
    T
    writetable(T,"/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/Dokumentaciok/Docs_CsutakB_PhD/11_Epid_MPC_Agent/actual/fig/Unexpected_results.xlsx");

end

%{
    set(0,'DefaultFigureWindowStyle','docked')
    for i = 1:numel(R)
        short_analysis(R{i});
    end
    for i = 1:numel(F)
        short_analysis(F{i});
    end
    
    set(0,'DefaultFigureWindowStyle','normal')
%}

%%

Names = {
    ... Alpha emerges
    ""  6000 "Free spread"
    ""  6000 "Parameters from Plot \textbf{A2} reused"
    ""  6000 "Weekly intervention planning"
    ""  6000 "Intervention planning every 2nd week"
    ""  6000 "Intervention planning every 3rd week"
    ""  6000 "Monthly intervention planning"
    ... Omicron emerges
    "" 10000 "Free spread"
    "" 10000 "Parameters from Plot \textbf{A2} reused"
    ""  5000 "Weekly intervention planning"
    ""  8000 "Intervention planning every 2nd week"
    ""  8000 "Intervention planning every 3rd week"
    ""  8000 "Monthly intervention planning"
    ... Omicron emerges just at the first day
    "" 16000 "Free spread"
    "" 16000 "Parameters from Plot \textbf{A2} reused"
    ""  5000 "Weekly intervention planning"
    ""  5000 "Intervention planning every 2nd week"
    ""  8000 "Intervention planning every 3rd week"
    "" 16000 "Monthly intervention planning"
    };

Plot_Width = 500;
Legend_Height = 190;
Plot_Height = 210;
FigDim = @(Idx) [Plot_Width Legend_Height+numel(Idx)*Plot_Height];

Idx = 2:6;
[fig] = Visualize(R,Names,Idx, ...
    "Xl_XData",C.Start_Date + 73,"Xl_Label","Alpha appears","Letter","E", ...
    "EmergesAt",70, ...
    "Tp",Tp,"Tcmp1",R{1}, ... "Tcmp2",T_Sc1_alpha, ...
    "FigNr",124,"FigDim",FigDim(Idx));
    % "FigNr",124,"FigDim",[500 870]);
exportgraphics(fig,fullfile(DIR_Summary,"Alpha70.pdf"));
% exportgraphics(fig,fullfile(DIR_Summary,"Alpha70.png"));

Idx = 8:12;
[fig] = Visualize(R,Names,Idx, ...
    "Xl_XData",C.Start_Date + 73,"Xl_Label","Omicron appears","Letter","F", ...
    "EmergesAt",70, ...
    "Tp",Tp,"Tcmp1",R{7}, ... "Tcmp2",T_fx1,
    "FigNr",123,"FigDim",FigDim(Idx));
    % "FigNr",123,"FigDim",[500 870]);
    % "FigNr",123,"FigDim",[500 700]);
exportgraphics(fig,fullfile(DIR_Summary,"Omicron70.pdf"));
% exportgraphics(fig,fullfile(DIR_Summary,"Omicron70.png"));


Idx = 14:18;
[fig] = Visualize(R,Names,Idx, ...
    "Letter","G","EmergesAt",1, ...
    "Tp",Tp,"Tcmp1",R{13}, ... "Tcmp2",T_Sc1_alpha, ...
    "FigNr",125,"FigDim",FigDim(Idx));
exportgraphics(fig,fullfile(DIR_Summary,"Omicron0.pdf"));
% exportgraphics(fig,fullfile(DIR_Summary,"Omicron0.png"));


% [fig,links] = Visualize(R,Names,[2,4],"Tp",Tp,"Tcmp1",T_fsp, ... "Tcmp2",T_fx1,
%     "FigNr",124);
% exportgraphics(fig,fullfile(DIR_Summary,"Summary24.pdf"));
% exportgraphics(fig,fullfile(DIR_Summary,"Summary24.png"));


function opts = detect(xls)
    opts = detectImportOptions(xls);

    Iq_k = string(opts.VariableNames(startsWith(opts.VariableNames,'Iq_')));
    SelVarNames = ["Date","TrRate","TrRateRec","Iref","L","P","A","I","H","Ir","TrRateBounds_2","NI",Iq_k];
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
    NI = zeros(N,n);
    AI = zeros(N,n);
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
        try
            Iq(:,:,i) = R(:,Iq_k).Variables;
        catch
            keyboard
        end
        
        I(:,i) = R.I;          % Active cases in the main phase with symptoms
        b(:,i) = R.TrRateRec;  % beta
        NI(:,i) = R.NI;        % New infections
        AI(:,i) = R.L + R.P + R.A + R.I + R.H; % All infected

        % An increase in the beta_multiplier detected
        if ismember("TrRateBounds_2",R.Properties.VariableNames)
            Idx = find(R.TrRateBounds_2(2:end) ./ R.TrRateBounds_2(1:end-1) > 1) + 1;
            IncMtp(Idx) = IncMtp(Idx) + 1;
        end
    end

    IncMtp = IncMtp ./ n;
    IncMtp(IncMtp < 0.2) = 0;


    Nt = height(R)-1;
    Knot_Density = 14;
    Nr_Knots = ceil(Nt / Knot_Density);
    Spline_Order = 5;
    
    sp = spap2(Nr_Knots,Spline_Order,0:Nt,mean(I,2));
    mean_I = fnval(sp,0:Nt);

    if ismember("Iref",R.Properties.VariableNames)
        R = R(:,"Iref");
    else
        R = R(:,[]);
        R.Iref = mean_I';
    end
    % --
    R.I = I;
    R.mean_I = mean(I,2);
    R.mean_I_sp = mean_I';
    R.std_I = std(I,[],2);
    R.IncMtp = IncMtp;
    % --
    [~,idx1] = min(vecnorm(I - R.Iref,2,1));
    R.closest_I = I(:,idx1);
    [~,idx2] = min(vecnorm(I - R.mean_I,2,1));
    R.typical_I = I(:,idx2);
    % --
    R.NI = NI;
    R.NA = AI;
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

    %%% 
    % Collect statistical information about the current set of measurements. These are
    % collected in the UserData of the returned table object.

    s = struct;

    s.DirName = dirname;
    s.db = n;
    s.Tp = Tp;

    s.Closest = xlsnames{idx1};
    s.ClosestIdx = idx1;
    s.Typical = xlsnames{idx2};
    s.TypicalIdx = idx2;

    CumInf = cumsum(mean(R.NI,2));
    CumInfPerc = @(Idx) round(CumInf(Idx)' / C.Np * 100,0);

    % Peaks
    [Peaks,PeaksOnDay] = findpeaks(R.mean_I_sp);
    ldx = Peaks < 150 | PeaksOnDay < 21;
    Peaks(ldx) = [];
    PeaksOnDay(ldx) = [];

    % Rebounds
    [Rebounds,ReboundsOnDay] = findpeaks(-R.mean_I_sp);
    ldx = -Rebounds < 150 | ReboundsOnDay < PeaksOnDay(1);
    Rebounds(ldx) = [];
    ReboundsOnDay(ldx) = [];

    pORr = [ ones(1,numel(Peaks)) , zeros(1,numel(Rebounds)) ];
    Extrema = round( [ Peaks(:) ; -Rebounds(:) ]' / C.Np * 100000 );
    ExtremaOnDay = [ PeaksOnDay(:) ; ReboundsOnDay(:) ]';
    
    [~,idx] = sort(ExtremaOnDay);
    pORr =  pORr(idx);
    Extrema = Extrema(idx);
    ExtremaOnDay = ExtremaOnDay(idx);

    s.Ci = CumInfPerc(numel(CumInf));

    p = 0;
    r = 0;
    for i = 1:numel(Extrema)
        if pORr(i) == 1
            p = p + 1;
            fn = "Peak" + num2str(p);
        else
            r = r + 1;
            fn = "Rbnd" + num2str(r);
        end

        s.(fn) = Extrema(i);
        s.(fn + "_Wk") = floor((ExtremaOnDay(i) - 1)/7)+1;
        s.(fn + "_Dy") = ExtremaOnDay(i) - 1;
        s.(fn + "_Ci") = CumInfPerc(ExtremaOnDay(i));
    end

    R.Properties.UserData = s;
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
    args.Xl_XData = [];
    args.Xl_LineSpec = 'k';
    args.Xl_Label = '';
    args.Letter = "A";
    args.EmergesAt = 70;
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
    plot(Okt23,[1,1],'LineWidth',2,'Color',Col.r_2,'DisplayName','~Prescribed flattened curve ($\mathbf{I}_k^{\mathrm{Ref}}$)~~');

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
    % patch(Okt23([1 2 2 1]),[1 1 2 2],Col.r_6, ...
    %         'EdgeColor',Col.r_6, ...
    %         'FaceAlpha',0.2, ...
    %         'EdgeAlpha',1, ...
    %         'LineWidth',1.5, ...
    %         'DisplayName','~Spread with flattening interventions of \textbf{A2}');

    plot(Okt23,[1,1],'Color',Color.Light_Gray,'DisplayName','~Multiple (50) simulations in PanSim');

    % xline(Okt23(1),'Color',COL.Color_Brown,'DisplayName','~(vertical) Increased transmission rate detected');

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

    counter = 0;
    for i = Idx
        counter = counter + 1;
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

        make_title_labeled(string(args.Letter) + num2str(counter),Names{i,3});
        
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
 
        R.IncMtp(1:args.EmergesAt) = 0;
        if any(R.IncMtp > 0) && i ~= Idx(1)
            Tply = cumsum(R.IncMtp(1:args.Tp(i):end)');
            idx = (find(Tply > 0.75,1) - 1) * args.Tp(i) + 1; 
            Xl = xline(R.Date(idx),'k','$\beta$ higher than expected', ...
                "LabelHorizontalAlignment","right","LabelOrientation","horizontal", ...
                "Interpreter","latex","FontSize",11,"HandleVisibility","off");
            % Xl(1).Label = 'detected';
            % Xl(1).LabelHorizontalAlignment = "right";
            % Xl(1).LabelOrientation = "horizontal";
            % Xl(1).Interpreter = "latex";
            % Xl(1).FontSize = 12;
        end

        ILim = [0,Names{i,2}];
        s = 1;
        YLineAt = 5000;

        ILim = ILim / 2;
        s = 100000 / C.Np;
        YLineAt = 2000;

        if ~isempty(args.Tcmp1)
            plot(args.Tcmp1.Date,args.Tcmp1.I*s,'Color',Color.Light_Gray,'HandleVisibility','off');
            Hide = plot_mean_var(args.Tcmp1.Date,args.Tcmp1.mean_I*s,args.Tcmp1.std_I*s,Col.r_5,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
            for pl = Hide'
                pl.HandleVisibility = 'off';
            end
        end
        
        if ~isempty(args.Tcmp2) && i == Idx(1)
            plot(args.Tcmp2.Date,args.Tcmp2.I*s,'Color',Color.Light_Gray,'HandleVisibility','off');
            Hide = plot_mean_var(args.Tcmp2.Date,args.Tcmp2.mean_I*s,args.Tcmp2.std_I*s,Col.r_6,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
            for pl = Hide'
                pl.HandleVisibility = 'off';
            end
        end
        
        if ~isempty(args.Xl_XData)
            xline(args.Xl_XData,args.Xl_LineSpec,args.Xl_Label, ...
                "LabelHorizontalAlignment","left","LabelOrientation","horizontal", ...
                "Interpreter","latex","FontSize",12,"HandleVisibility","off");
        end
        
        plot(R.Date,R.I*s,'Color',Color.Light_Gray,'HandleVisibility','off');
    
        Hide = plot_mean_var(R.Date,R.mean_I*s,R.std_I*s,Col.r_3,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
        for pl = Hide'
            pl.HandleVisibility = 'off';
        end
        plot(R.Date,R.typical_I*s,'Color',Color.Black,'HandleVisibility','off');
    
        plot(Date,R.Iref*s,'LineWidth',2,'Color',Col.r_2,'HandleVisibility','off');
     
        yline(YLineAt,'Color',Col.r_2,'LineWidth',1);
        ylim(ILim)
    
        ax.TickLabelInterpreter = "latex";
        ax.FontSize = 12;
        
        ax.XGrid = 'off';
        ax.XMinorGrid = 'off';
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

function short_analysis(A)
%%
    figure, nexttile, hold on;
    plot(A.Date,A.I,'Color',Color.Light_Gray,'HandleVisibility','off');

    Hide = plot_mean_var(A.Date,A.mean_I,A.std_I,Col.r_3,"Alpha",2,"LineWidth",2,"PlotLim",false,"FaceAlpha",0.5);
    for pl = Hide'
        pl.HandleVisibility = 'off';
    end
    plot(A.Date,A.typical_I,'Color',Color.Black,'HandleVisibility','off');

    plot(A.Date,A.Iref,'LineWidth',2,'Color',Col.r_2,'HandleVisibility','off');

    if ~isempty(A.Properties.UserData.PeaksOnDay)
        xline(A.Date(1) + A.Properties.UserData.PeaksOnDay{1},'k','peak')
    end

    if ~isempty(A.Properties.UserData.ReboundsOnDay)
        xline(A.Date(1) + A.Properties.UserData.ReboundsOnDay{1},'k','rbd')
    end

    [~,bname,~] = fileparts(A.Properties.UserData.DirName);

    title(strrep(bname,'_',' '))

    A.Properties.UserData
end
