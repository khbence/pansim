%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

%%

% Population in the simulator
Np = C.Np;

xlsnames = dir('/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Allcomb_2024-03-07/*.xls');
idx = randperm(numel(xlsnames),1);
idx = 323;
xls = xlsnames(idx);

R = readtimetable(fullfile(xls.folder,xls.name));
R.Date.Format = 'uuuu-MM-dd';

Start_Date = datetime(2020,10,01);
% End_Date = Start_Date + 300;
End_Date = datetime(2021,01,31);

R = R(isbetween(R.Date,Start_Date,End_Date),:);

%%%

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(string(fp.dir) + "/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q(["Transient","Original","Future"],:);

% Q("Original","Period_L") = table(2);
% Q("Original","Period_P") = table(4);
% Q("Original","Period_A") = table(7);
% Q("Original","Pr_D") = table(0.48);

% Itt szandekosan elrontom a parametereket
Q("Original","Pr_I") = table(0.7);
Q("Original","Period_L") = table(5.5);
Q("Original","Period_P") = table(10.1);
Q("Original","Period_A") = table(10);
Q("Original","Pr_D") = table(0.1);


Q("Original","Period_L") = table(1.8);


% 2024.02.26. (február 26, hétfő), 15:06
Q("Original","Pr_I") = table(0.46);
% Q("Original","Period_L") = table(1.5);
% Q("Original","Period_P") = table(3.1);
% Q("Original","Period_A") = table(4);
% Q("Original","Pr_D") = table(1.2);


P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,Start_Date,End_Date),:);
P = hp.param2table(P.Param);
R(:,P.Properties.VariableNames) = P;

K = Epid_Par.GetK;

R = rec_SLPIAHDR(R,"Visualize",true);

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