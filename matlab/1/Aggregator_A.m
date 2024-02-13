%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 05. (2023a)
%


DIR = "/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results";
Directories = dir(fullfile(DIR,"*_Finalized"));

[~,bname,~] = fileparts(mfilename("fullpath"));
Aggregator_DIR = fullfile(DIR,bname);

if ~exist(Aggregator_DIR,"dir")
    mkdir(Aggregator_DIR)
end

PM = load_policy_measures;

xls = fullfile(DIR,Directories(1).name,"A.xls");
opts = detectImportOptions(xls);
opts = setvartype(opts,policy_varnames,"categorical");
opts = setvartype(opts,policy_varnames,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts.SelectedVariableNames = ["Date" policy_varnames simout_varnames "TrRate" "Ioff"];

for i = 2:length(Directories)
    Old_Path = fullfile(DIR,Directories(i).name,"Fig.pdf");
    New_Path = fullfile(Aggregator_DIR,"Fig_" + Directories(i).name(8:23) + ".pdf");

    if contains(Directories(i).name,"LUT")
        fprintf("Update LUT: %d/%d\n",i,length(Directories))
    end

    copyfile(Old_Path,New_Path);

    Old_Path = fullfile(DIR,Directories(i).name,"A.xls");
    New_Path = fullfile(Aggregator_DIR,"A_" + Directories(i).name(8:23) + ".xls");
    copyfile(Old_Path,New_Path);
end

