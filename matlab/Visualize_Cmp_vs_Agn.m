%%
% Author: Peter Polcz (ppolcz@gmail.com) 
% Created on 2023. August 22. (2023a)

%%

fp = pcz_mfilename(mfilename("fullpath"));
DIR = fp.dir + "/Results";
if ~exist(DIR,"dir")
    mkdir(DIR)
end
DIR = DIR + "/";

SAVE_DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/COVID-Elorejelzesek/Megmentett_Eletek/";

Plot_Colors


%%

% A 120 fő/km² feletti népsűrűségű, városias településeken lakók aránya
% https://www.ksh.hu/stadat_files/fol/hu/fol0008.html
Varosiasodas = 62.29;

% https://www.ksh.hu/stadat_files/nep/hu/nep0003.html
Osszlakossag_2020 = 9769526;

% Agens-alapu szimulátor ágenseinek száma
Nr_Agent = 179500;

% A 120 fő/km² feletti népsűrűségű, városias településeken lakók száma
Varosban_lakok = round(Osszlakossag_2020 * Varosiasodas / 100);

%% Load data

agent = load(DIR+"REC_2021-07-20_Agent-based.mat");
agent_Sc0 = load(DIR+"REC_2021-07-01_Agent-based_Sc0_PLNONE_CFNONE_SONONE_QU0_Ujra");
agent_Sc1 = load(DIR+"REC_2021-07-01_Agent-based_Sc1_CFNONE_SONONE_Ujra.mat");
agent_Sc2 = load(DIR+"REC_2021-07-01_Agent-based_Sc2_CFNONE_Ujra.mat");
agent_Sc3 = load(DIR+"REC_2021-07-01_Agent-based_Sc3_QU0_Ujra.mat");
agent_D21 = load(DIR+"REC_2021-07-01_Agent-based_Delay21.mat");
cmp = load(DIR+"REC_2023-07-23_CmpWW-based__2.mat");

Ta = agent.T;
Ta0 = agent_Sc0.T;
Ta1 = agent_Sc1.T;
Ta2 = agent_Sc2.T;
Ta3 = agent_Sc3.T;
Ta21 = agent_D21.T;
Tc = cmp.T;

Ts = {Ta0 , Ta1 , Ta2 , Ta3};
nT = numel(Ts);

Mtp_Agent = Varosban_lakok / Nr_Agent;
Mtp_Comp = Varosban_lakok / Tc.Np(1);

%% Visualize

fig = figure(123);
fig.Position(3:4) = [691 1056];

Sorrend = [3 2 4 1];

XLim = [Ta.Date(1) datetime(2021,06,01)];

Lsty_Agn = {'LineWidth',3,'Color',C_.C49};
Lsty_Cmp = {'LineWidth',2,'Color',C_.C31};
Lsty = {
    'LineWidth',1,'LineStyle','--','Color',C_.C1
    'LineWidth',1.5,'LineStyle',':','Color',C_.C29  
    'LineWidth',1,'LineStyle','-.','Color',C_.C4  
    'LineWidth',1,'LineStyle','--','Color',C_.C51  
    };

Tl = tiledlayout(4,1,"TileSpacing","loose","Padding","tight");
Ax = nexttile; grid on, box on, hold on
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    TeX("\textbf{Rekonstruált} járványgörbék a valóságban alkalmazott intézkedésekkel:"))
Pl_aI = plot(Ta.Date,Ta.I * Mtp_Agent,Lsty_Agn{:},'DisplayName', ...
    TeX("~~rekonstrukció ágens alapú modell segítségével"));
plot(Tc.Date,(Tc.P + Tc.I + Tc.A + Tc.H) * Mtp_Comp,Lsty_Cmp{:},'DisplayName', ...
    TeX('~~rekonstrukció kompartmentális modell segítségével'))
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    TeX("\textbf{Becsült} járványgörbék, ha a következő enyhítéseket vezettük volna be:"))
for i = Sorrend
    switch i
        case 1 % "Sc0"
            % Szórakozóhelyeket nem zárjuk be
            % Nincs kijárási korlátozás
            % Iskolákat nem zárjuk be
            % Nincs karantén
            Policy_measures = [ "PLNONE" "CFNONE" "SONONE" "QU0" ];
        case 2 % "Sc1"
            % Nincs kijárási korlátozás
            % Iskolákat nem zárjuk be
            Policy_measures = [ "CFNONE" "SONONE" ];
        case 3 % "Sc2"
            % Nincs kijárási korlátozás
            Policy_measures = [ "CFNONE" ];
        case 4 % "Sc3"
            % Nincs karantén
            Policy_measures = [ "QU0" ];
    end
    Policy_measures_str = strrep(strjoin(Policy_measures,", "),".","");
    DpName = "\makebox[6.3cm]{~~" + Policy_measures_str + "\hfill}: " + num2str(round((Ts{i}.D1(end)-Ta.D1(end))*Mtp_Agent)) + " megmentett \'elet";

    plot(Ts{i}.Date,Ts{i}.I * Mtp_Agent,Lsty{i,:},'DisplayName',DpName)
end
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    "\makebox[1cm]{Ahol:}~--\makebox[2cm]{~PLNONE:\hfill} sz\'orakoz\'ohelyeket nem z\'arjuk be")
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    "\makebox[1cm]{}~--\makebox[2cm]{~CFNONE:\hfill} nincs kij\'ar\'asi korl\'atoz\'as")
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    "\makebox[1cm]{}~--\makebox[2cm]{~SONONE:\hfill} iskol\'akat nem z\'arjuk be")
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    "\makebox[1cm]{}~--\makebox[2cm]{~QU0:\hfill} nincs karant\'en")
jf = java.text.DecimalFormat;
plot(Tc.Date(1)-1,0,'w','DisplayName', ...
    TeX([ newline ...
    '(A becslések az ágens alapú modell segítségével a 120 fő/km$^2$ feletti népsűrűségű' newline ...
    '~\,városias településeken lakó, összesen \textbf{' strrep(char(jf.format(Varosban_lakok)),',','.') '} lélekszámra lettek elvégezve.)']))
% plot(Tc.Date(1)-1,0,'w','DisplayName', ...
%     "(A g\""orb\'ek a 120 f\H{o}/km$^2$ feletti n\'eps\H{u}r\H{u}s\'eg\H{u}, " + ...
%     "v\'arosias telep\""ul\'eseken lak\'o,")
% plot(Tc.Date(1)-1,0,'w','DisplayName', ...
%     "~\,\""osszesen \textbf{" + strrep(char(jf.format(Varosban_lakok)),',','.') + ...
%     "} l\'eleksz\'amra vannak normaliz\'alva.)")
plot(Ta.Date,Ta.I * Mtp_Agent,Lsty_Agn{:},'HandleVisibility','off')

cm = "9.8cm";
Cnt(double('A')-1);

Leg = legend(Interpreter="latex",FontSize=12,Location="northoutside");
Leg.Box = 'off';
title("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Fert\H{o}z\""ottek sz\'ama (SEIR modell eset\'en az `I')\hfill}","FontSize",13,"Interpreter","latex")

Ax = [ Ax nexttile ]; grid on, box on, hold on
Pl_aB = plot(Ta.Date,movmean(Ta.TrRate,14),Lsty_Agn{:});
plot(Tc.Date,Tc.TrRate,Lsty_Cmp{:});
for i = Sorrend
    plot(Ts{i}.Date,movmean(Ts{i}.TrRate,14),Lsty{i,:})
end
plot(Ta.Date,movmean(Ta.TrRate,14),Lsty_Agn{:});
title("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} V\'irus terjed\'es\'enek sebess\'ege\hfill}","FontSize",13,"Interpreter","latex")

Ax = [ Ax nexttile ]; grid on, box on, hold on
Pl_aD = plot(Ta.Date,Ta.D1 * Mtp_Agent,Lsty_Agn{:});
plot(Tc.Date,Tc.D * Mtp_Comp,Lsty_Cmp{:});
for i = Sorrend
    plot(Ts{i}.Date,Ts{i}.D1 * Mtp_Agent,Lsty{i,:})
end
plot(Ta.Date,Ta.D1 * Mtp_Agent,Lsty_Agn{:})
title("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Elhunytak sz\'ama\hfill}","FontSize",13,"Interpreter","latex")

Ax = [ Ax nexttile ]; grid on, box on, hold on
Pl_aD = plot(Ta.Date,(Ta.I5h + Ta.I6h) * Mtp_Agent,Lsty_Agn{:});
plot(Tc.Date,Tc.H_off_ma * Mtp_Comp,Lsty_Cmp{:})
for i = Sorrend
    plot(Ts{i}.Date,(Ts{i}.I5h + Ts{i}.I6h) * Mtp_Agent,Lsty{i,:})
end
Sh = plot_interval(XLim,[0 0],[0.8 1]*1e4,C_.C31,"PlotLim",false);
Pl_aD = plot(Ta.Date,(Ta.I5h + Ta.I6h) * Mtp_Agent,Lsty_Agn{:});
title("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} K\'orh\'azban \'apoltak sz\'ama\hfill}","FontSize",13,"Interpreter","latex")
Leg = legend(Sh(3),TeX(['Becsült elméleti felső' newline 'határ a kórházi terhelésre']),Interpreter="latex",FontSize=12,Location="northeast",Box='off');
Leg.Box = 'off';

for ax = Ax
    ax.TickLabelInterpreter = "latex";
    ax.FontSize = 12;
    
    ax.XTick = Ta.Date(day(Ta.Date) == 1 & mod(month(Ta.Date),2)==0);
    ax.XMinorGrid = 'on';
    ax.XAxis.MinorTick = 'off';
    ax.XAxis.MinorTickValues = Ta.Date(weekday(Ta.Date) == 1); 
    ax.XLim = XLim;
end

clipboard("copy",SAVE_DIR)
exportgraphics(fig,SAVE_DIR + "agens_specialis_intezkedesek.png")
exportgraphics(fig,SAVE_DIR + "agens_specialis_intezkedesek.pdf","ContentType","vector")



