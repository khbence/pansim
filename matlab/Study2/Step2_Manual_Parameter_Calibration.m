function Step2_Manual_Parameter_Calibration
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Revised on 2025. May 08. (2024b)

fp = pcz_mfilename(mfilename("fullpath"));
ff = @(d) string(cellfun(@(s) {fullfile(s.folder,s.name)}, num2cell(d)));
dirname = fullfile(fp.dir,"Output/OpenLoop_Simulations");
xlsnames = ff( dir(fullfile(dirname,"*.xls")) );

for i = 1:10

Nr = ceil(rand(1)*numel(xlsnames));
xls = xlsnames(Nr);
R = readtimetable(xls);
R.Date.Format = 'uuuu-MM-dd';

Start_Date = datetime(2020,10,01);
End_Date = datetime(2021,01,31);

R = R(isbetween(R.Date,Start_Date,End_Date),:);
R.IQ = Vn.IQ(R);

%%

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(string(fp.pdir) + "/Parameters/Par_HUN_2024-02-26_Agens_Wild.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q(["Transient","Original","Alpha"],:);

% Water Research cikkben levo parameterek
Q("Original","Period_I") = table(3);
Q("Original","Period_L") = table(3);
Q("Original","Period_P") = table(4);
Q("Original","Period_A") = table(4);
Q("Original","Period_H") = table(12);
Q("Original","Pr_I") = table(0.46);
Q("Original","Pr_H") = table(0.076);
Q("Original","Pr_D") = table(0.259);

P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,Start_Date,End_Date),:);
P = hp.param2table(P.Param);
R(:,P.Properties.VariableNames) = P;
R1 = rec_SLPIAHDR(R,"Visualize",false,'PWConstBeta',false);

% PLOS CB cikkben levo parameterek
% 2024.08.08. (augusztus  8, csütörtök), 18:02
Q("Original","Period_I") = table(4);
Q("Original","Pr_I") = table(0.48);
Q("Original","Period_L") = table(1.5);
Q("Original","Period_P") = table(3.1);
Q("Original","Period_A") = table(4);
Q("Original","Pr_H") = table(0.076);
Q("Original","Period_H") = table(12);
Q("Original","Pr_D") = table(0.48);

display(rows2vars(Q("Original",5:end-4)),'Parameters')

P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,Start_Date,End_Date),:);
P = hp.param2table(P.Param);
R(:,P.Properties.VariableNames) = P;
R2 = rec_SLPIAHDR(R,"Visualize",false,'PWConstBeta',false);

%%

[~,~,~,J] = epid.ode_model_8comp;

x_PanSim = R(:,Vn.SLPIAHDR).Variables;
x1 = R1(:,Vn.SLPIAHDR + "r").Variables;
x2 = R2(:,Vn.SLPIAHDR + "r").Variables;

Fig = figure(61512);
delete(Fig.Children)
Fig.Position(3:4) = [500 500];
Tl = tiledlayout(3,3,"TileSpacing","compact","Padding","tight");

Ax = nexttile; hold on, grid on, box on;
plot(R.Date,[R.TrRate , R1.TrRateRec , R2.TrRateRec],'LineWidth',1.2)
title('$\beta$','Interpreter','latex','FontSize',12)
Ax.XTickLabels = {};

[~,simname] = fileparts(xls);
for i = 1:J.nx
    Ax = [Ax , nexttile]; hold on, grid on, box on
    Pl = plot(R.Date,[x_PanSim(:,i) , x1(:,i) , x2(:,i)],'LineWidth',1.2);
    title("\textbf{" + Vn.SLPIAHDR(i) + "}",'Interpreter','latex','FontSize',12)

    if ~ismember(Vn.SLPIAHDR(i),["H" "D" "R"])
        Ax(end).XTickLabels = {};
    end

    % Legend
    if Vn.SLPIAHDR(i) == "S"
        Pl(1).DisplayName = "Simulator's output (" + simname + ".xls)";
        Pl(2).DisplayName = "Reconstruction using parameters from previous work~~~~~";
        Pl(3).DisplayName = "Reconstruction using calibrated parameters";
        legend('Location','northoutside','Box','off','Interpreter','latex','FontSize',13)
    end
end

for ax = Ax
    ax.TickLabelInterpreter = "latex";
    ax.FontSize = 12;
end

exportgraphics(Fig,fullfile(fp.dir,"Output","Model_Fit_" + simname + ".jpg"))


end