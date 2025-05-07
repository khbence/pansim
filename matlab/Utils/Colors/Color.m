classdef Color
    
    properties (Constant)
    
        Plot_1 = [0 0.4470 0.7410];
        Plot_2 = [0.8500 0.3250 0.0980];
        Plot_3 = [0.9290 0.6940 0.1250];
        Plot_4 = [0.4940 0.1840 0.5560];
        Plot_5 = [0.4660 0.6740 0.1880];
        Plot_6 = [0.3010 0.7450 0.9330];
        Plot_7 = [0.6350 0.0780 0.1840];
        Black           = pcz_hex2rgb("#000000");  %  8
        Light_Black     = pcz_hex2rgb("#808080");  %  9
        Blue            = pcz_hex2rgb("#0000FF");  % 10
        Dark_Blue       = pcz_hex2rgb("#00008B");  % 11
        Light_Blue      = pcz_hex2rgb("#ADD8E6");  % 12
        Brown           = pcz_hex2rgb("#A52A2A");  % 13
        Dark_Brown      = pcz_hex2rgb("#5C4033");  % 14
        Light_Brown     = pcz_hex2rgb("#996600");  % 15
        Buff            = pcz_hex2rgb("#F0DC82");  % 16
        Dark_Buff       = pcz_hex2rgb("#976638");  % 17
        Light_Buff      = pcz_hex2rgb("#ECD9B0");  % 18
        Cyan            = pcz_hex2rgb("#00FFFF");  % 19
        Dark_Cyan       = pcz_hex2rgb("#008B8B");  % 20
        Light_Cyan      = pcz_hex2rgb("#E0FFFF");  % 21
        Gold            = pcz_hex2rgb("#FFD700");  % 22
        Dark_Gold       = pcz_hex2rgb("#EEBC1D");  % 23
        Light_Gold      = pcz_hex2rgb("#F1E5AC");  % 24
        Goldenrod       = pcz_hex2rgb("#DAA520");  % 25
        Dark_Goldenrod  = pcz_hex2rgb("#B8860B");  % 26
        Light_Goldenrod = pcz_hex2rgb("#FFEC8B");  % 27
        Gray            = pcz_hex2rgb("#808080");  % 28
        Dark_Gray       = pcz_hex2rgb("#404040");  % 29
        Light_Gray      = pcz_hex2rgb("#D3D3D3");  % 30
        Green           = pcz_hex2rgb("#008000");  % 31
        Dark_Green      = pcz_hex2rgb("#006400");  % 32
        Light_Green     = pcz_hex2rgb("#90EE90");  % 33
        Ivory           = pcz_hex2rgb("#FFFFF0");  % 34
        Dark_Ivory      = pcz_hex2rgb("#F2E58F");  % 35
        Light_Ivory     = pcz_hex2rgb("#FFF8C9");  % 36
        Magenta         = pcz_hex2rgb("#FF00FF");  % 37
        Dark_Magenta    = pcz_hex2rgb("#8B008B");  % 38
        Light_Magenta   = pcz_hex2rgb("#FF77FF");  % 39
        Mustard         = pcz_hex2rgb("#FFDB58");  % 40
        Dark_Mustard    = pcz_hex2rgb("#7C7C40");  % 41
        Light_Mustard   = pcz_hex2rgb("#EEDD62");  % 42
        Orange          = pcz_hex2rgb("#FFA500");  % 43
        Dark_Orange     = pcz_hex2rgb("#FF8C00");  % 44
        Light_Orange    = pcz_hex2rgb("#D9A465");  % 45
        Pink            = pcz_hex2rgb("#FFC0CB");  % 46
        Dark_Pink       = pcz_hex2rgb("#E75480");  % 47
        Light_Pink      = pcz_hex2rgb("#FFB6C1");  % 48
        Red             = pcz_hex2rgb("#FF0000");  % 49
        Dark_Red        = pcz_hex2rgb("#8B0000");  % 50
        Light_Red       = pcz_hex2rgb("#FF3333");  % 51
        Silver          = pcz_hex2rgb("#C0C0C0");  % 52
        Dark_Silver     = pcz_hex2rgb("#AFAFAF");  % 53
        Light_Silver    = pcz_hex2rgb("#E1E1E1");  % 54
        Turquoise       = pcz_hex2rgb("#30D5C8");  % 55
        Dark_Turquoise  = pcz_hex2rgb("#00CED1");  % 56
        Light_Turquoise = pcz_hex2rgb("#AFE4DE");  % 57
        Violet          = pcz_hex2rgb("#EE82EE");  % 58
        Dark_Violet     = pcz_hex2rgb("#9400D3");  % 59
        Light_Violet    = pcz_hex2rgb("#7A5299");  % 60
        White           = pcz_hex2rgb("#FFFFFF");  % 61
        Yellow          = pcz_hex2rgb("#FFFF00");  % 62
        Dark_Yellow     = pcz_hex2rgb("#FFCC00");  % 63
        Light_Yellow    = pcz_hex2rgb("#FFFFE0");  % 64

        Colors = ([
            Color.Plot_1            %  1
            Color.Plot_2            %  2
            Color.Plot_3            %  3
            Color.Plot_4            %  4
            Color.Plot_5            %  5
            Color.Plot_6            %  6
            Color.Plot_7            %  7
            Color.Blue              %  8
            Color.Brown             %  9
            Color.Buff              % 10
            Color.Cyan              % 11
            Color.Gold              % 12
            Color.Dark_Blue         % 13
            Color.Light_Blue        % 14
            Color.Dark_Brown        % 15
            Color.Light_Brown       % 16
            Color.Dark_Buff         % 17
            Color.Light_Buff        % 18
            Color.Dark_Cyan         % 19
            Color.Light_Cyan        % 20
            Color.Dark_Gold         % 21
            Color.Light_Gold        % 22
            Color.Goldenrod         % 23
            Color.Dark_Goldenrod    % 24
            Color.Light_Goldenrod   % 25
            Color.Gray              % 26
            Color.Dark_Gray         % 27
            Color.Light_Gray        % 28
            Color.Green             % 29
            Color.Dark_Green        % 30
            Color.Light_Green       % 31
            Color.Ivory             % 32
            Color.Dark_Ivory        % 33
            Color.Light_Ivory       % 34
            Color.Magenta           % 35
            Color.Dark_Magenta      % 36
            Color.Light_Magenta     % 37
            Color.Mustard           % 38
            Color.Dark_Mustard      % 39
            Color.Light_Mustard     % 40
            Color.Orange            % 41
            Color.Dark_Orange       % 42
            Color.Light_Orange      % 43
            Color.Pink              % 44
            Color.Dark_Pink         % 45
            Color.Light_Pink        % 46
            Color.Red               % 47
            Color.Dark_Red          % 48
            Color.Light_Red         % 49
            Color.Silver            % 50
            Color.Dark_Silver       % 51
            Color.Light_Silver      % 52
            Color.Turquoise         % 53
            Color.Dark_Turquoise    % 54
            Color.Light_Turquoise   % 55
            Color.Violet            % 56
            Color.Dark_Violet       % 57
            Color.Light_Violet      % 58
            Color.White             % 59
            Color.Yellow            % 60
            Color.Dark_Yellow       % 61
            Color.Light_Yellow      % 62
            Color.Black             % 63
            Color.Light_Black       % 64
            ]);
    end

    methods(Static)
        function plotColors
            
            fig = figure(41231);
            fig.Position(3) = 1960;
            Tl = tiledlayout(1,1,'Padding','compact');
            nexttile; hold on;
        
            for r=1:height(Color.Colors)
                x = linspace(0,r,500);
                y = sqrt(r.^2-x.^2);
                plot(x,y,'LineWidth',15,'Color',Color.Colors(r,:))
            end
        
            grid on
            ylim([0,0.1])
            xlim([0,height(Color.Colors)])
            xticks(1:height(Color.Colors))
        
        end
    end

end
