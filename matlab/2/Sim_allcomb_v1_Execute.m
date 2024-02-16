%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)
% 

% fp = pcz_mfilename(mfilename('fullpath'));
% DIR_Data = fullfile(fp.pdir,'Data');
% 
% filename = fullfile(DIR_Data,'data2.txt');
% savename = fullfile(DIR_Data,'Policy_measures.mat');
% T = hp.load_policy_measures('filename',filename,'savename',savename,'reset',true);
% T(isnan(T.Pmx),:) = [];

T = hp.load_policy_measures;

nP = max(T.Pmx);

%%

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2/Result_" + string(Now) + "_T" + num2str(Tp) + "_allcomb";
mkdir(DIR)

%%

Tp = 28;

N = 6*7*4;
Nr_Periods = N / Tp;

PM_Indices = zeros(nP,Nr_Periods);
PM_Indices(:,1) = randperm(nP)';
for i = 2:Nr_Periods
    PM_Indices(:,i) = circshift(PM_Indices(:,i-1),7+i);
end

Actual_Pmx = 0;

save('matlab/2/PM_Indices.mat',"DIR","Actual_Pmx","nP","PM_Indices","Tp","N","Nr_Periods")

for i = 1:nP
    Sim_allcomb_v1
end

delete('matlab/2/PM_Indices.mat')

