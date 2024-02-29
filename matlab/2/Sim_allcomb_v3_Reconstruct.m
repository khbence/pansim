%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
% 
% `Idx` should be set first

fp = pcz_mfilename(mfilename("fullpath"));
DIR = fullfile(fp.dir,'Output','Allcomb_2024-02-27');

ff = @(d) string(cellfun(@(s) {fullfile(s.folder,s.name)}, num2cell(d)));
xlsnames = ff( dir(fullfile(DIR,'*.xls')) );

opts = detectImportOptions(xlsnames(1));
opts = setvartype(opts,opts.SelectedVariableNames,"double");
opts = setvartype(opts,Vn.policy,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");
opts = setvartype(opts,"IQ","int32");

T = readtimetable(xlsnames(1),opts,'Sheet','Results');

Q = readtable(fullfile(fp.pdir,"Parameters","Par_HUN_2024-02-26_Agens_Wild.xlsx"), ...
    "ReadRowNames",true,"Sheet","Main");
P = Epid_Par.Get(Q);
P = P(isbetween(P.Date,T.Date(1),T.Date(end)),:);
P = hp.param2table(P.Param);

%%


N = height(T)*numel(xlsnames);

na = nan(N,1);
D = table(na,na,'VariableNames',["IQ","TrRateRec"]);
D(1,:) = [];

idx = 1:height(T);
for i = 1:length(xlsnames)

    T = readtimetable(xlsnames(i),opts,'Sheet','Results');

    try
        Pr = readtimetable(xlsnames(i),'Sheet','Reconstruction');
        TPr = [T,Pr];
    catch e
        if strcmp(e.identifier,'MATLAB:spreadsheet:book:openSheetName')
            
            TP = [T,P];
            [TPr,NewVars] = rec_SLPIAHDR(TP);
            Variables = [P.Properties.VariableNames , NewVars];        
            
            writetimetable(TPr(:,Variables),xlsnames(i),'Sheet','Reconstruction','WriteMode','overwritesheet')
        else
            MException(e.identifier,"[Pcz: unhandled exception] " + e.message)
        end
    end
    
    D(idx,:) = TPr(:,D.Properties.VariableNames);
    idx = idx + height(T);
end

D.IQ = int32(D.IQ);
D = renamevars(D,"TrRateRec","TrRate");

GS = groupsummary(D,"IQ",["mean","std"]);



