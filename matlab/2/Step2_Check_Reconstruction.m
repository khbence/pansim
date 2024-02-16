%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)

%%

% Population in the simulator
Np = C.Np;

RESULT = "Result_2024-02-13_16-59_T28_allcomb";
R = readtimetable("/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/" + RESULT + "/A47.xls");
R.Date.Format = 'uuuu-MM-dd';

Start_Date = datetime(2020,10,01);
% End_Date = Start_Date + 300;
End_Date = datetime(2021,01,31);

R = R(isbetween(R.Date,Start_Date,End_Date),:);

%%

R = rec_SLPIAHDR(R);

%%

fp = pcz_mfilename(mfilename('fullpath'));
Q = readtable(string(fp.dir) + "/Parameters/Par_HUN_2023-12-19_JN1.xlsx", ...
    "ReadRowNames",true,"Sheet","Main");
Q = Q(["Transient","Original","Future"],:);

P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,Start_Date,End_Date),:);

K = Epid_Par.GetK;

%% Construct optimization

import casadi.*

[f,~,~,J] = epid.ode_model_8comp(Np);

x = R(:,Vn.SLPIAHDR + "r").Variables';
p = P.Param';
beta = R.TrRateRec;

model_error = x*0;

% Enforce the state equations
for i = 1:height(R)-1
    x_kp1 = full(f.Fn(x(:,i),p(:,i),beta(i),0,0));
    model_error(:,i) = x_kp1 - x(:,i+1);
end
