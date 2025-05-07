classdef Col < COL

    properties (Constant)
        r_1  = COL.Color_1;
        r_2  = COL.Color_2;
        r_3  = COL.Color_3;
        r_4  = COL.Color_4;
        r_5  = COL.Color_5;
        r_6  = COL.Color_6;
        r_7  = COL.Color_7;
        r_8  = COL.Color_Black;
        r_9  = COL.Color_Light_Black;
        r_10 = COL.Color_Blue;
        r_11 = COL.Color_Dark_Blue;
        r_12 = COL.Color_Light_Blue;
        r_13 = COL.Color_Brown;
        r_14 = COL.Color_Dark_Brown;
        r_15 = COL.Color_Light_Brown;
        r_16 = COL.Color_Buff;
        r_17 = COL.Color_Dark_Buff;
        r_18 = COL.Color_Light_Buff;
        r_19 = COL.Color_Cyan;
        r_20 = COL.Color_Dark_Cyan;
        r_21 = COL.Color_Light_Cyan;
        r_22 = COL.Color_Gold;
        r_23 = COL.Color_Dark_Gold;
        r_24 = COL.Color_Light_Gold;
        r_25 = COL.Color_Goldenrod;
        r_26 = COL.Color_Dark_Goldenrod;
        r_27 = COL.Color_Light_Goldenrod;
        r_28 = COL.Color_Gray;
        r_29 = COL.Color_Dark_Gray;
        r_30 = COL.Color_Light_Gray;
        r_31 = COL.Color_Green;
        r_32 = COL.Color_Dark_Green;
        r_33 = COL.Color_Light_Green;
        r_34 = COL.Color_Ivory;
        r_35 = COL.Color_Dark_Ivory;
        r_36 = COL.Color_Light_Ivory;
        r_37 = COL.Color_Magenta;
        r_38 = COL.Color_Dark_Magenta;
        r_39 = COL.Color_Light_Magenta;
        r_40 = COL.Color_Mustard;
        r_41 = COL.Color_Dark_Mustard;
        r_42 = COL.Color_Light_Mustard;
        r_43 = COL.Color_Orange;
        r_44 = COL.Color_Dark_Orange;
        r_45 = COL.Color_Light_Orange;
        r_46 = COL.Color_Pink;
        r_47 = COL.Color_Dark_Pink;
        r_48 = COL.Color_Light_Pink;
        r_49 = COL.Color_Red;
        r_50 = COL.Color_Dark_Red;
        r_51 = COL.Color_Light_Red;
        r_52 = COL.Color_Silver;
        r_53 = COL.Color_Dark_Silver;
        r_54 = COL.Color_Light_Silver;
        r_55 = COL.Color_Turquoise;
        r_56 = COL.Color_Dark_Turquoise;
        r_57 = COL.Color_Light_Turquoise;
        r_58 = COL.Color_Violet;
        r_59 = COL.Color_Dark_Violet;
        r_60 = COL.Color_Light_Violet;
        r_61 = COL.Color_White;
        r_62 = COL.Color_Yellow;
        r_63 = COL.Color_Dark_Yellow;
        r_64 = COL.Color_Light_Yellow;
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
