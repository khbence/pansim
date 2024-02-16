

PanSim_Results = '/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2';
Result = 'Result_2024-02-13_16-59_T28_allcomb';

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
DIR = fullfile(PanSim_Results,Result + "_GenLUT");
mkdir(DIR)

ff = @(d) string(cellfun(@(s) {fullfile(s.folder,s.name)}, num2cell(d)));
xlsnames = [
    ff( dir(fullfile(PanSim_Results,Result,'*.xls')) )
    ff( dir(fullfile(PanSim_Results,"*_Finalized")) ) + "/A.xls"
    ];

T = hp.load_policy_measures;
T = T(:,[Vn.policy "Pmx"]);

opts = detectImportOptions(xlsnames(1));
opts = setvartype(opts,opts.SelectedVariableNames,"double");
opts = setvartype(opts,Vn.policy,"categorical");
opts = setvartype(opts,"Date","datetime");
opts = setvaropts(opts,"Date","DatetimeFormat","yyyy-MM-dd");

D = Read(xlsnames(1),opts,T,DIR,1);
for i = 2:length(xlsnames)
    Di = Read(xlsnames(i),opts,T,DIR,i);

    D = [ D ; Di ];
end

function D = Read(xls,opts,T,DIR,idx)
    D = readtimetable(xls,opts);
    D.(Vn.policy_Iq) = D(:,Vn.policy_Iq_).Variables;
    D.TrRateBounds = [D.TrRateBounds_1 D.TrRateBounds_2];
    D(:,[Vn.policy_Iq_,"TrRateBounds_1","TrRateBounds_2"]) = [];
    
    D = join(D,T);
    Tp = find(abs(diff(D.Pmx)),1);

    D = rec_SLPIAHDR(D,'WeightBetaSlope',1e4);
    writetimetable(D,fullfile(DIR,sprintf("A%03d.xls",idx)))

    % fig = Visualize_MPC(D,height(D),Tp=Tp);
    % drawnow
    % exportgraphics(fig,fullfile(DIR,sprintf("Fig%03d.jpg",D.Pmx(1))))
end