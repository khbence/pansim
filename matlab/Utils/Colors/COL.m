classdef COL
    
    properties (Constant)
    
        Color_1 = [0 0.4470 0.7410];
        Color_2 = [0.8500 0.3250 0.0980];
        Color_3 = [0.9290 0.6940 0.1250];
        Color_4 = [0.4940 0.1840 0.5560];
        Color_5 = [0.4660 0.6740 0.1880];
        Color_6 = [0.3010 0.7450 0.9330];
        Color_7 = [0.6350 0.0780 0.1840];
        Color_Black           = pcz_hex2rgb("#000000");  %  8
        Color_Light_Black     = pcz_hex2rgb("#808080");  %  9
        Color_Blue            = pcz_hex2rgb("#0000FF");  % 10
        Color_Dark_Blue       = pcz_hex2rgb("#00008B");  % 11
        Color_Light_Blue      = pcz_hex2rgb("#ADD8E6");  % 12
        Color_Brown           = pcz_hex2rgb("#A52A2A");  % 13
        Color_Dark_Brown      = pcz_hex2rgb("#5C4033");  % 14
        Color_Light_Brown     = pcz_hex2rgb("#996600");  % 15
        Color_Buff            = pcz_hex2rgb("#F0DC82");  % 16
        Color_Dark_Buff       = pcz_hex2rgb("#976638");  % 17
        Color_Light_Buff      = pcz_hex2rgb("#ECD9B0");  % 18
        Color_Cyan            = pcz_hex2rgb("#00FFFF");  % 19
        Color_Dark_Cyan       = pcz_hex2rgb("#008B8B");  % 20
        Color_Light_Cyan      = pcz_hex2rgb("#E0FFFF");  % 21
        Color_Gold            = pcz_hex2rgb("#FFD700");  % 22
        Color_Dark_Gold       = pcz_hex2rgb("#EEBC1D");  % 23
        Color_Light_Gold      = pcz_hex2rgb("#F1E5AC");  % 24
        Color_Goldenrod       = pcz_hex2rgb("#DAA520");  % 25
        Color_Dark_Goldenrod  = pcz_hex2rgb("#B8860B");  % 26
        Color_Light_Goldenrod = pcz_hex2rgb("#FFEC8B");  % 27
        Color_Gray            = pcz_hex2rgb("#808080");  % 28
        Color_Dark_Gray       = pcz_hex2rgb("#404040");  % 29
        Color_Light_Gray      = pcz_hex2rgb("#D3D3D3");  % 30
        Color_Green           = pcz_hex2rgb("#008000");  % 31
        Color_Dark_Green      = pcz_hex2rgb("#006400");  % 32
        Color_Light_Green     = pcz_hex2rgb("#90EE90");  % 33
        Color_Ivory           = pcz_hex2rgb("#FFFFF0");  % 34
        Color_Dark_Ivory      = pcz_hex2rgb("#F2E58F");  % 35
        Color_Light_Ivory     = pcz_hex2rgb("#FFF8C9");  % 36
        Color_Magenta         = pcz_hex2rgb("#FF00FF");  % 37
        Color_Dark_Magenta    = pcz_hex2rgb("#8B008B");  % 38
        Color_Light_Magenta   = pcz_hex2rgb("#FF77FF");  % 39
        Color_Mustard         = pcz_hex2rgb("#FFDB58");  % 40
        Color_Dark_Mustard    = pcz_hex2rgb("#7C7C40");  % 41
        Color_Light_Mustard   = pcz_hex2rgb("#EEDD62");  % 42
        Color_Orange          = pcz_hex2rgb("#FFA500");  % 43
        Color_Dark_Orange     = pcz_hex2rgb("#FF8C00");  % 44
        Color_Light_Orange    = pcz_hex2rgb("#D9A465");  % 45
        Color_Pink            = pcz_hex2rgb("#FFC0CB");  % 46
        Color_Dark_Pink       = pcz_hex2rgb("#E75480");  % 47
        Color_Light_Pink      = pcz_hex2rgb("#FFB6C1");  % 48
        Color_Red             = pcz_hex2rgb("#FF0000");  % 49
        Color_Dark_Red        = pcz_hex2rgb("#8B0000");  % 50
        Color_Light_Red       = pcz_hex2rgb("#FF3333");  % 51
        Color_Silver          = pcz_hex2rgb("#C0C0C0");  % 52
        Color_Dark_Silver     = pcz_hex2rgb("#AFAFAF");  % 53
        Color_Light_Silver    = pcz_hex2rgb("#E1E1E1");  % 54
        Color_Turquoise       = pcz_hex2rgb("#30D5C8");  % 55
        Color_Dark_Turquoise  = pcz_hex2rgb("#00CED1");  % 56
        Color_Light_Turquoise = pcz_hex2rgb("#AFE4DE");  % 57
        Color_Violet          = pcz_hex2rgb("#EE82EE");  % 58
        Color_Dark_Violet     = pcz_hex2rgb("#9400D3");  % 59
        Color_Light_Violet    = pcz_hex2rgb("#7A5299");  % 60
        Color_White           = pcz_hex2rgb("#FFFFFF");  % 61
        Color_Yellow          = pcz_hex2rgb("#FFFF00");  % 62
        Color_Dark_Yellow     = pcz_hex2rgb("#FFCC00");  % 63
        Color_Light_Yellow    = pcz_hex2rgb("#FFFFE0");  % 64

        Colors = ([
            COL.Color_1                 %  1
            COL.Color_2                 %  2
            COL.Color_3                 %  3
            COL.Color_4                 %  4
            COL.Color_5                 %  5
            COL.Color_6                 %  6
            COL.Color_7                 %  7
            COL.Color_Blue              %  8
            COL.Color_Brown             %  9
            COL.Color_Buff              % 10
            COL.Color_Cyan              % 11
            COL.Color_Gold              % 12
            COL.Color_Dark_Blue         % 13
            COL.Color_Light_Blue        % 14
            COL.Color_Dark_Brown        % 15
            COL.Color_Light_Brown       % 16
            COL.Color_Dark_Buff         % 17
            COL.Color_Light_Buff        % 18
            COL.Color_Dark_Cyan         % 19
            COL.Color_Light_Cyan        % 20
            COL.Color_Dark_Gold         % 21
            COL.Color_Light_Gold        % 22
            COL.Color_Goldenrod         % 23
            COL.Color_Dark_Goldenrod    % 24
            COL.Color_Light_Goldenrod   % 25
            COL.Color_Gray              % 26
            COL.Color_Dark_Gray         % 27
            COL.Color_Light_Gray        % 28
            COL.Color_Green             % 29
            COL.Color_Dark_Green        % 30
            COL.Color_Light_Green       % 31
            COL.Color_Ivory             % 32
            COL.Color_Dark_Ivory        % 33
            COL.Color_Light_Ivory       % 34
            COL.Color_Magenta           % 35
            COL.Color_Dark_Magenta      % 36
            COL.Color_Light_Magenta     % 37
            COL.Color_Mustard           % 38
            COL.Color_Dark_Mustard      % 39
            COL.Color_Light_Mustard     % 40
            COL.Color_Orange            % 41
            COL.Color_Dark_Orange       % 42
            COL.Color_Light_Orange      % 43
            COL.Color_Pink              % 44
            COL.Color_Dark_Pink         % 45
            COL.Color_Light_Pink        % 46
            COL.Color_Red               % 47
            COL.Color_Dark_Red          % 48
            COL.Color_Light_Red         % 49
            COL.Color_Silver            % 50
            COL.Color_Dark_Silver       % 51
            COL.Color_Light_Silver      % 52
            COL.Color_Turquoise         % 53
            COL.Color_Dark_Turquoise    % 54
            COL.Color_Light_Turquoise   % 55
            COL.Color_Violet            % 56
            COL.Color_Dark_Violet       % 57
            COL.Color_Light_Violet      % 58
            COL.Color_White             % 59
            COL.Color_Yellow            % 60
            COL.Color_Dark_Yellow       % 61
            COL.Color_Light_Yellow      % 62
            COL.Color_Black             % 63
            COL.Color_Light_Black       % 64
            ]);
    end

    methods(Static)
        function plotColors
            
            fig = figure(41231);
            fig.Position(3) = 1960;
            Tl = tiledlayout(1,1,'Padding','compact');
            nexttile; hold on;
        
            for r=1:height(COL.Colors)
                x = linspace(0,r,500);
                y = sqrt(r.^2-x.^2);
                plot(x,y,'LineWidth',15,'Color',COL.Colors(r,:))
            end
        
            grid on
            ylim([0,0.1])
            xlim([0,height(COL.Colors)])
            xticks(1:height(COL.Colors))
        
        end
    end

end
