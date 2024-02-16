function T = load_policy_measures_2

fp = pcz_mfilename(mfilename('fullpath'));

Idx = find(strcmp(fp.dirs,'matlab'));
DIR_Data = [filesep , fullfile(fp.dirs{end:-1:Idx},'Data')];

filename = fullfile(DIR_Data,'Policy_measures_2.xls');

opts = detectImportOptions(filename);
opts = setvartype(opts,Vn.policy,"categorical");

T = readtable(filename,opts);
T = hp.quantify_policy(T);

end