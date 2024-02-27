%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)
% 

fp = pcz_mfilename(mfilename("fullpath"));
DIR_Export = fullfile(fp.dir,'Export');
MAT_PM_Allcomb = fullfile(DIR_Export,'PM_Allcomb.mat');

Today = string(datetime('today','Format','uuuu-MM-dd'));
DIR = fullfile(DIR_Export,"Allcomb_" + Today);

if ~exist(DIR,"dir")
    mkdir(DIR);
end

%%

T = Vn.allcomb;
nP = height(T);

% [Logikai] Ha nincs karantén, akkor nem érdemes intenzívebben tesztelni
% T(T.QU_Val == 0 & T.TP_Val > 1,:) = [];

% [Politikai] Ha az iskolat valamilyen mértékben bezárjuk, akkor a szórakozóhelyet is
% zárjuk
% T(T.SO_Val > 0 & T.PL_Val == 0,:) = [];

Tp = 28;
Nr_Periods = 13;
N = Tp * Nr_Periods;

N_sim = 1000;

Pmx_Perms = zeros(N_sim,Nr_Periods);
for i = 1:N_sim
    Pmx_Perms(i,:) = randperm(nP,Nr_Periods);
end

allperms = Pmx_Perms(:);

fig = figure(12);
histogram(allperms-0.5,'BinEdges',(0:nP));

Idx = 0;
save(MAT_PM_Allcomb,"DIR","T","nP","Idx","Pmx_Perms","Tp","N","Nr_Periods")

for i = 1:nP
    Sim_allcomb_v3
end

delete(MAT_PM_Allcomb)

