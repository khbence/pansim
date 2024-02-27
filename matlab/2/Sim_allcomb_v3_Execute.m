%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
%  Revised on 2023. October 16. (2023a)
% 

fp = pcz_mfilename(mfilename("fullpath"));
DIR_Input = fullfile(fp.dir,'Input');
DIR_Output = fullfile(fp.dir,'Output');
MAT_PM_Allcomb = fullfile(DIR_Input,'PM_Allcomb.mat');
MAT_PM_combi = @(i) fullfile(DIR_Input,sprintf('PM_comb%04d.mat',i));

Today = string(datetime('today','Format','uuuu-MM-dd'));
DIR = fullfile(DIR_Output,"Allcomb_" + Today);

if ~exist(DIR_Input,"dir")
    mkdir(DIR_Input)
end

if ~exist(DIR_Output,"dir")
    mkdir(DIR_Output)
end

if ~exist(DIR,"dir")
    mkdir(DIR);
end

%% PM_Allcomb.mat

T = Vn.allcomb;
nP = height(T);

% [Logikai] Ha nincs karantén, akkor nem érdemes intenzívebben tesztelni
% T(T.QU_Val == 0 & T.TP_Val > 1,:) = [];

% [Politikai] Ha az iskolat valamilyen mértékben bezárjuk, akkor a szórakozóhelyet is
% zárjuk
% T(T.SO_Val > 0 & T.PL_Val == 0,:) = [];

Tp = 28;
Nr_Periods = 13;
save(MAT_PM_Allcomb,"DIR","T","Tp","Nr_Periods")

%% PM_comb000i.mat

N_sim = 1000;
Pmx_Perms = zeros(N_sim,Nr_Periods);
for i = 1:N_sim
    Perm = randperm(nP,Nr_Periods);
    Pmx_Perms(i,:) = Perm;
    save(MAT_PM_combi(i),"Perm")
end

allperms = Pmx_Perms(:);

fig = figure(12);
histogram(allperms-0.5,'BinEdges',(0:nP));
