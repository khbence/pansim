%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
% 
% `Idx` should be set first

fp = pcz_mfilename(mfilename("fullpath"));
Result_ID = "2024-02-27";
DIR = fullfile(fp.dir,"Output","Allcomb_" + Result_ID);

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


VariableNames = ["NewCases","NewCasesRate","Rt","IQ","TrRateRec"];
na = repmat({nan(N,1)},size(VariableNames));
D = table(na{:},'VariableNames',VariableNames);

idx = 1:height(T);
for i = 1:length(xlsnames)

    T = readtimetable(xlsnames(i),opts,'Sheet','Results');

    try
        Pr = readtimetable(xlsnames(i),'Sheet','Reconstruction');
        T = [T,Pr];
    catch e
        if strcmp(e.identifier,'MATLAB:spreadsheet:book:openSheetName')
            
            TP = [T,P];
            [T,NewVars] = rec_SLPIAHDR(TP);
            Variables = [P.Properties.VariableNames , NewVars];        
            
            writetimetable(T(:,Variables),xlsnames(i),'Sheet','Reconstruction','WriteMode','overwritesheet')
        else
            MException(e.identifier,"[Pcz: unhandled exception] " + e.message)
        end
    end

    T = renamevars(T,"NI","NewCases");

    Knot_Density = 28;
    Nr_Knots = height(T) / Knot_Density;
    Spline_Order = 5;
    dt = 0.1;

    Day = days(T.Date - T.Date(1));
    NI_sp = spap2(Nr_Knots,Spline_Order,Day,T.NewCases);
    dNI_sp = fnder(NI_sp);

    T.NewCasesRate = fnval(dNI_sp,Day);

    T.Rt = T.TrRateRec .* (1./T.tauL + T.pI./T.tauI + T.qA.*(1 - T.pI)./T.tauA).*T.S/C.Np; 
    
    D(idx,:) = T(:,D.Properties.VariableNames);
    idx = idx + height(T);

    fprintf('%d / %d\n',i,length(xlsnames))
end

D.IQ = int32(D.IQ);
D = renamevars(D,"TrRateRec","TrRate");

% figure, hold on, plot(D.NewCases / C.Np * 10000), plot(D.NewCasesRate)

% Filter out if new cases are small
D(D.NewCases ./ C.Np * 100000 < 10 , :) = [];
D(D.NewCasesRate < 0,:) = [];

GS = groupsummary(D,"IQ",["mean","std","median"]);
GS = renamevars(GS,["mean_TrRate","std_TrRate","median_TrRate"],["TrRate","TrRateStd","TrRateMedian"]);

s = load(fullfile(fp.dir,"Input","PM_Allcomb.mat"));
GS = join(s.T,GS);

[~,idx] = sort(GS.TrRate);
GS = GS(idx,:);

fname = fullfile(fp.dir,"Output","Summary_" + Result_ID + ".xls");
writetable(GS,fname);

%%

fname = '/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output/Summary_2024-02-27.xls';
% fname = '/home/ppolcz/_PanSim_HEAD/matlab/2/Output/Summary_2024-02-27.xls';
opts = detectImportOptions(fname);
opts = setvartype(opts,Vn.policy,"categorical");
GS = readtable(fname,opts);

Iq_k = GS.Properties.VariableNames(startsWith(GS.Properties.VariableNames,'Iq_'));
GS.Iq = GS(:,Iq_k).Variables;
GS(:,Iq_k) = [];

GS.Pmx = (1:height(GS))';

Visualize_Intervention_Simple(GS,TrRateStd="TrRateStd",TrRateMedian="TrRateMedian",FigNr=1231);

