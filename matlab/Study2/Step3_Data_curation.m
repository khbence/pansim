function Step3_Data_curation
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Revised on 2025. May 07. (2024b)

fp = pcz_mfilename(mfilename("fullpath"));

ff = @(d) string(cellfun(@(s) {fullfile(s.folder,s.name)}, num2cell(d)));
dirname = fullfile(fp.dir,"Output");
xlsnames = ff( dir(fullfile(dirname,"Allcomb_*",'*.xls')) );

Today = string(datetime('today','Format','uuuu-MM-dd'));
recdir = fullfile(dirname,"AllCombRec_" + Today);
if ~exist(recdir,"dir")
    mkdir(recdir)
end

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

Z = zeros(height(T),numel(xlsnames));
I_all = Z;
Ir_all = Z;

VariableNames = ["NewCases","NewCasesRate","Rt","IQ","TrRateRec",Vn.policy + "_Val"];
na = repmat({nan(N,1)},size(VariableNames));
D = table(na{:},'VariableNames',VariableNames);

idx = 1:height(T);
for i = 1:length(xlsnames)
    tic

    T = readtimetable(xlsnames(i),opts,'Sheet','Results');
    T = Vn.quantify_policy(T);
    TP = [T,P];
    [T,NewVars] = rec_SLPIAHDR(TP,'PWConstBeta',true);
    Variables = [P.Properties.VariableNames , NewVars];

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

    I_all(:,i) = T.I;
    Ir_all(:,i) = T.Ir;

    %%% ---------

    ldx = T.Ir ./ C.Np * 100000 < 10;
    
    o = ones(height(R),1);
    z = o; z(ldx) = NaN;
    z = fillmissing(z,"next");
    
    ldx = ismissing(z);

    T.Iref = max(T.I) * ones(height(R),1);
    T.Iref(ldx) = 0;
    T.TrRateRange = ones(height(R),1) * [0 0.5];
    T.TrRateRange(ldx,:) = NaN;

    fig = Visualize_MPC_v3(T,N+1,Nr_Periods,"Tp",max(Tp,7)); drawnow
    writetimetable(T(:,Variables),xlsnames(i),'Sheet','Reconstruction','WriteMode','overwritesheet')
    exportgraphics(fig,strrep(xlsnames(i),".xls",".jpg"));
    fprintf('%d / %d\n',i,length(xlsnames))

    % fullfile(recdir,"Pmx" + sprintf('_%d',T.Pmx(find(diff(T.Pmx)))) + "_i" + num2str(i) );
    toc
end

D.IQ = int32(D.IQ);
D = renamevars(D,"TrRateRec","TrRate");

% figure, hold on, plot(D.NewCases / C.Np * 10000), plot(D.NewCasesRate)

% Filter out if new cases are small
D(D.NewCases ./ C.Np * 100000 < 10 , :) = [];
% D(D.NewCasesRate < 0,:) = [];

GS = groupsummary(D,"IQ",["mean","std","median"]);
GS = renamevars(GS,["mean_TrRate","std_TrRate","median_TrRate"],["TrRate","TrRateStd","TrRateMedian"]);

s = load(fullfile(fp.dir,"Input","PM_Allcomb.mat"));
GS = join(s.T,GS);

% Teljes lezaras
[~,idx] = max(GS.IQ);
GS.TrRate(idx) = min(GS.TrRate) - 1e-5;

% Teljes felengedes
[~,idx] = min(GS.IQ);
GS.TrRate(idx) = max(GS.TrRate) + 1e-5;

[~,idx] = sort(GS.TrRate);
GS = GS(idx,:);

fname = fullfile(dirname,"Summary_" + string(datetime('today','Format','uuuu-MM-dd')) + ".xls");
writetable(GS,fname);

%%

Date = T.Date;
fname = fullfile(dirname,"Summary_" + string(datetime('today','Format','uuuu-MM-dd')) + "_Iall.mat");
save(fname,'Date','Ir_all','I_all');

plot(T.Date,I_all)

%%

fname = 'PanSim_Output/Summary_2024-03-14.xls';
opts = detectImportOptions(fname);
opts = setvartype(opts,Vn.policy,"categorical");
GS = readtable(fname,opts);

Iq_k = GS.Properties.VariableNames(startsWith(GS.Properties.VariableNames,'Iq_'));
GS.Iq = GS(:,Iq_k).Variables;
GS(:,Iq_k) = [];

GS.Pmx = (1:height(GS))';

size(GS)
summary(GS.TrRateStd,'all')

GS(GS.QU == "QU2",:) = [];

TrRateStd = GS.TrRateStd;
GS.TrRateStd = movmean(TrRateStd,5);
Visualize_Intervention_Simple3(GS,TrRateStd="TrRateStd",FigNr=1230);
GS.TrRateStd = TrRateStd;
exportgraphics(gcf,"Fig1230.png")

[~,Idx] = sort(GS.CF,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.PL,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.MA,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.QU,'descend'); GS = GS(Idx,:);

TrRateStd = GS.TrRateStd;
GS.TrRateStd = movmean(TrRateStd,5);
Visualize_Intervention_Simple3(GS,TrRateStd="TrRateStd",FigNr=1231);
GS.TrRateStd = TrRateStd;
exportgraphics(gcf,"Fig1231.png")

GS.TP(GS.TP == "TPdef") = "TP05";
GS.TP(GS.TP == "TP015") = "TP15";
GS.TP(GS.TP == "TP035") = "TP35";

[~,Idx] = sort(GS.TP,'descend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.SO,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.CF,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.PL,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.MA,'ascend'); GS = GS(Idx,:);
[~,Idx] = sort(GS.QU,'descend'); GS = GS(Idx,:);

TrRateStd = GS.TrRateStd;
GS.TrRateStd = movmean(TrRateStd,5);
Visualize_Intervention_Simple3(GS,TrRateStd="TrRateStd",FigNr=1232);
GS.TrRateStd = TrRateStd;
exportgraphics(gcf,"Fig1232.png")

