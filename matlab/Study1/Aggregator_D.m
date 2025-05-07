function [D,Db] = Aggregator_D(T)
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 07. (2023a)
%

DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results";
Directories = dir(fullfile(DIR,"*_Finalized"));

fp = pcz_mfilename(mfilename("fullpath"));
Aggregator_DIR = fullfile(DIR,fp.bname);

if ~exist(Aggregator_DIR,"dir")
    mkdir(Aggregator_DIR)
end

xls = fullfile(DIR,Directories(1).name,"A.xls");
opts = detectImportOptions(xls);
opts = setvartype(opts,policy_varnames,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts.SelectedVariableNames = ["Date" policy_varnames simout_varnames "TrRate"];

xlsname = @(i) fullfile(DIR,Directories(i).name,"A.xls");
D = readtimetable(xlsname(1),opts);
for i = 2:length(Directories)
    Path = fullfile(DIR,Directories(i).name,"A.xls");
    Di = readtimetable(Path,opts);

    D = [ D ; Di ];
end
D(ismissing(D.TrRate),:) = [];
D(D.TrRate < 0.01,:) = [];

% Recompute the transmission rate
[~,D.TrRate] = get_SLPIAb(D(:,simout_varnames).Variables,'ImmuneIdx',1:7);

%%

D = join(D,T(:,[policy_varnames , "Idx" "Intelligent" "Beta"]));
D.TrRateMtp = D.TrRate ./ D.Beta;

% Find out the relative multiplier between the LUT's beta and the measurements.
Db = retime(D(:,"TrRateMtp"),'weekly','mean');
Db = retime(Db,'daily','spline');

for i = 1:days(D.Date(end) - Db.Date(end))
    
end

end
