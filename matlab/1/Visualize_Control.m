%%
% Author: Peter Polcz (ppolcz@gmail.com) 
% Created on 2023. August 22. (2023a)

%%

% Agens-alapu szimulátor ágenseinek száma
Nr_Agent = 179500;

fp = pcz_mfilename(mfilename("fullpath"));
DIR = fp.dir + "/Results";
if ~exist(DIR,"dir")
    mkdir(DIR)
end
DIR = DIR + "/";

% Available in _Epid/Utils
Plot_Colors

LOAD_DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/";
SAVE_DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/COVID-Elorejelzesek/Megmentett_Eletek/";

T = readtable(LOAD_DIR + 'Result_2023-08-05_16:55.xls');

simout0 = T(:,startsWith(T.Properties.VariableNames,"simout0")).Variables;
simout1 = T(:,startsWith(T.Properties.VariableNames,"simout1")).Variables;
    
S1 = simout2table(simout1);

TP = table(["TPdef"; "TP015"; "TP035"],[0.5; 1.5; 3.5],'VariableNames',{'TP','TP_Val'});
PL = table(["PLNONE";"PL0"],[0;1],'VariableNames',{'PL','PL_Val'});
CF = table(["CFNONE";"CF2000-0500"],[0;1],'VariableNames',{'CF','CF_Val'});
SO = table(["SONONE";"SO3";"SO12"],[0;1;2],'VariableNames',{'SO','SO_Val'});
QU = table(["QU0";"QU2";"QU3"],[0;2;4],'VariableNames',{'QU','QU_Val'});
MA = table(["MA0.8";"MA1.0"],[0.8;1],'VariableNames',{'MA','MA_Val'});

T = join(join(join(join(join(join(T,TP),PL),CF),SO),QU),MA);

% TP = [ TP015, TP035, TPdef ]    
% 1.5%-os, 3.5% illetve azt hiszem 0.5% átlagos testingje a népességnek
% 
% PL = [ PL0, PLNONE ]    
% lezárod-e a szórakozó helyeket vagy sem
% 
% CF = [ CF2000-0500, CFNONE ]    
% van e 20-5 curfew
% 
% SO = [ SO12, SO3, SONONE ]    
% 12. osztályig zársz le sulit, 3. osztályig, nincs suli lezárás
% 
% QU = [ QU0, QU2, QU3 ]         
% nincs karantén, te és veled együttlakók bezárása, QU2 + munkahely zárás
% 
% MA = [ MA0.8, MA1.0 ]      
% itt a szám hogy a fertőzés valószínűség mennyivel van maszk által normálva
% 
% TP a tesztelési mennyiség:
% TPdef ami Magyarországon volt úgy kb, 
% TP015, ha a lakosság 1.5%-át teszteljük naponta, 
% TP035 meg ha 3.5%-át
% 
% SO12 hogy csak 12 év alattiak járnak iskolaba, 
% SO3 hogy senki, 
% SONONE hogy mindenki jár
% 
% CF2000-0500 ami volt kijárási korlátozás este 8-reggel 5
% 
% MA0.8 rádob 0.8-as szorzót a fertőzésre kültéri vagy maszkviselős helyszíneken
% 
% QU0 senki nincs karanténozva (diagnosztizált se), 
% QU2 a diagnosztizált és családja, 
% QU3 pedig ha még osztálya, munkatársai is

%%

cm = "11.6cm";
Cnt(double('A')-1);


fig = figure(123);
fig.Position(3) = 1453;
fig.Position(4) = 756;
Tl = tiledlayout(6,2,"TileSpacing","compact","Padding","compact","TileIndexing","columnmajor");

Ax = nexttile([3,1]); hold on, grid on, box on;
Pl = plot(S1.Date,T.Iref,'Color',Color_2, ...
    'DisplayName',TeX("Előírt görbe"));
Pl = [Pl , plot(S1.Date,S1.I,'Color',Color_1, ...
    'DisplayName',TeX("Szimulált görbe"))];
Leg = legend(Interpreter="latex",FontSize=12,Location="northwest");
title("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Fert\H{o}z\""ottek sz\'ama egy "+num2str(Nr_Agent)+TeX(" lélekszámú városban \hfill}"),"FontSize",13,"Interpreter","latex")
Ax.YAxis.Exponent = 0;

Ax = [Ax nexttile([3,1])]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.simbeta)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Járvány terjedésének sebessége ($\beta$)\hfill}"),"FontSize",13,"Interpreter","latex")

Ax = [Ax nexttile]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.TP_Val)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Tesztelési intenzitás (\% lakosság)-ban\hfill}"),"FontSize",13,"Interpreter","latex")
ylabel(TeX("\% lakosság"),Interpreter="latex")
yticks([0.5,1.5,3.5])

Ax = [Ax nexttile]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.PL_Val)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Szórakozó helyeket lezárása vagy sem\hfill}"),"FontSize",13,"Interpreter","latex")
yticks([0,1]), ylim([-0.2,1.2]), yticklabels(["NEM","IGEN"])

Ax = [Ax nexttile]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.CF_Val)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Kijárási korlátozás este 8 és reggel 5 között\hfill}"),"FontSize",13,"Interpreter","latex")
yticks([0,1]), ylim([-0.2,1.2]), yticklabels(["NEM","IGEN"])

Ax = [Ax nexttile]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.SO_Val)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Iskolazárás\hfill}"),"FontSize",13,"Interpreter","latex")
yticks([0,1,2]), ylim([-0.4,2.4]), yticklabels(["NINCS","3.o-ig","12.o-ig"])

Ax = [Ax nexttile]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.QU_Val)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Karantén irányelv szigorúsága\hfill}"),"FontSize",13,"Interpreter","latex")
yticks([0,2,4]), ylim([-0.4,4.4]), yticklabels(["NINCS",TeX("lakótársak"),"lkt+mhely"])

Ax = [Ax nexttile]; hold on, grid on, box on;
Pl = [ Pl , plot(S1.Date,T.MA_Val)];
title(TeX("\makebox["+cm+"]{\textbf{"+char(Cnt)+".} Maszkviselés hatékonysága, mint a járvány terjedését lassító együttható\hfill}"),"FontSize",13,"Interpreter","latex")
yticks([0.8,1]), ylim([0.75,1.05])

XLim = [S1.Date([1,end])];

set(Pl,'LineWidth',1.5);

for ax = Ax
    ax.TickLabelInterpreter = "latex";
    ax.FontSize = 12;
    
    ax.XTick = S1.Date(day(S1.Date) == 1 & mod(month(S1.Date),2)==0);
    ax.XMinorGrid = 'on';
    ax.XAxis.MinorTick = 'off';
    ax.XAxis.MinorTickValues = S1.Date(weekday(S1.Date) == 1); 
    ax.XLim = XLim;
end

clipboard("copy",SAVE_DIR)
exportgraphics(fig,SAVE_DIR + "agens_Ctrl.png")
exportgraphics(fig,SAVE_DIR + "agens_Ctrl.pdf","ContentType","vector")



