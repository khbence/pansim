%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)
%  Revised on 2024. April 12. (2023a)

%%

% Population in the simulator
Np = C.Np;

RESULT = "Result_2024-02-13_16-59_T28_allcomb";
xls = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/" + RESULT + "/A47.xls";
xls = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/GenLUT/Fig_2024-02-26_14-28.xls";
R = readtimetable(xls);
R.Date.Format = 'uuuu-MM-dd';

Start_Date = datetime(2020,10,01);
% End_Date = Start_Date + 300;
End_Date = datetime(2021,01,31);

R = R(isbetween(R.Date,Start_Date,End_Date),:);
R.IQ = Vn.IQ(R); % Update: 2024-04-12

%%

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(string(fp.dir) + "/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q(["Transient","Original","Future"],:);

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
drawnow 

% Q("Original","Period_L") = table(2);
% Q("Original","Period_P") = table(4);
% Q("Original","Period_A") = table(7);
% Q("Original","Pr_D") = table(0.48);

% 2024.02.26. (február 26, hétfő), 15:06
Q("Original","Period_I") = table(4);
Q("Original","Pr_I") = table(0.46);
Q("Original","Period_L") = table(1.5);
Q("Original","Period_P") = table(3.1);
Q("Original","Period_A") = table(4);
Q("Original","Pr_H") = table(0.09);
Q("Original","Period_H") = table(10);
Q("Original","Pr_D") = table(0.42);

% 2024.08.08. (augusztus  8, csütörtök), 18:02 -- cikkben
Q("Original","Period_I") = table(4);
Q("Original","Pr_I") = table(0.48);
Q("Original","Period_L") = table(1.5);
Q("Original","Period_P") = table(3.1);
Q("Original","Period_A") = table(4);
Q("Original","Pr_H") = table(0.076);
Q("Original","Period_H") = table(12);
Q("Original","Pr_D") = table(0.48);


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
Tl = tiledlayout(3,3,"TileSpacing","compact","Padding","tight")

Ax = nexttile; hold on, grid on, box on;
plot(R.Date,[R.TrRate , R1.TrRateRec , R2.TrRateRec],'LineWidth',1.2)
title('$\beta$','Interpreter','latex','FontSize',12)
Ax.XTickLabels = {};

for i = 1:J.nx
    Ax = [Ax , nexttile]; hold on, grid on, box on
    Pl = plot(R.Date,[x_PanSim(:,i) , x1(:,i) , x2(:,i)],'LineWidth',1.2);
    title("\textbf{" + Vn.SLPIAHDR(i) + "}",'Interpreter','latex','FontSize',12)

    if ~ismember(Vn.SLPIAHDR(i),["H" "D" "R"])
        Ax(end).XTickLabels = {};
    end

    % Legend
    if Vn.SLPIAHDR(i) == "S"
        Pl(1).DisplayName = "Simulator's output";
        Pl(2).DisplayName = "Reconstruction using parameters from previous work~~~~~";
        Pl(3).DisplayName = "Reconstruction using calibrated parameters";
        legend('Location','northoutside','Box','off','Interpreter','latex','FontSize',13)
    end
end

for ax = Ax
    ax.TickLabelInterpreter = "latex";
    ax.FontSize = 12;
end

exportgraphics(Fig,['/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/Dokumentaciok/Docs_CsutakB_PhD/11_Epid_MPC_Agent/actual/fig/' ...
    'ParamCalib.pdf'],'ContentType','vector')

% exportgraphics(gcf,['/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/Dokumentaciok/Docs_CsutakB_PhD/13_UNKP/fig/' ...
%     'ParamEst8.pdf'],'ContentType','vector')

return

%% Construct optimization

import casadi.*

[f,~,~,J] = epid.ode_model_8comp(Np);

x = R(:,Vn.SLPIAHDR + "r").Variables';
p = R(:,Vn.params).Variables';
beta = R.TrRateRec;

model_error = x*0;

% Enforce the state equations
for i = 1:height(R)-1
    x_kp1 = full(f.Fn(x(:,i),p(:,i),beta(i),0,0));
    model_error(:,i) = x_kp1 - x(:,i+1);
end
