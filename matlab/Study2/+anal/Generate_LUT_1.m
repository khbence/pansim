
ff = @(d) string(cellfun(@(s) {fullfile(s.folder,s.name)}, num2cell(d)));
xlsnames = ff( dir(fullfile(C.DIR_GenLUT,'*.xls')) );

T = hp.load_policy_measures;
T = T(:,[Vn.policy "Pmx"]);

D = Read(xlsnames(1));
for i = 2:length(xlsnames)
    Di = Read(xlsnames(i));
    D = [ D ; Di ];
end

GS = groupsummary(D(:,["Pmx","TrRate"]),"Pmx",["mean","std","min","max","median"]);

%%

[~,idx] = unique(D.Pmx);
Du = timetable2table(D(idx,["Pmx",Vn.policy]));
Du.Date = [];
GS = join(GS,Du);

GS = addvars(GS,GS.mean_TrRate,'After',"Pmx",'NewVariableNames',"Beta");
GS = movevars(GS,Vn.policy,'After',"Pmx");

%%

fp = pcz_mfilename(mfilename('fullpath'));

Idx = find(strcmp(fp.dirs,'matlab'));
DIR_Data = [filesep , fullfile(fp.dirs{end:-1:Idx},'Data')];

savename = fullfile(DIR_Data,'Policy_measures_2.xls');
writetable(GS,savename);

function D = Read(xls)
    persistent opts

    [dir,bname,ext] = fileparts(xls);

    if isempty(opts)
        opts = detect(xls);
    end

    D = readtimetable(xls,opts);
    fprintf('Reading %s%s\n',bname,xls)

    if ~all(ismember(opts.SelectedVariableNames(2:end),D.Properties.VariableNames))
        opts = detect(xls);
        D = readtimetable(xls,opts);

        fprintf('Again: %s%s\n',bname,xls)
    end

    % D.(Vn.policy_Iq) = D(:,Vn.policy_Iq_).Variables;
    % D.TrRateBounds = [D.TrRateBounds_1 D.TrRateBounds_2];
    % D(:,[Vn.policy_Iq_,"TrRateBounds_1","TrRateBounds_2"]) = [];
    
    D.Pmx(2:end) = D.Pmx(1:end-1);
    D(1,:) = [];
end

function opts = detect(xls)
    opts = detectImportOptions(xls);
    opts = setvartype(opts,opts.SelectedVariableNames,"double");
    opts = setvartype(opts,Vn.policy,"categorical");
    opts = setvartype(opts,"Date","datetime");
    opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
    opts.SelectedVariableNames = ["Date","TrRate","Pmx",Vn.policy];
end
