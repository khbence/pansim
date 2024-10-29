function R = Sim_interventions_by_xls(xlsname,PanSim_args,DirName,Name,args)
arguments
    xlsname,PanSim_args,DirName,Name
    args.Visualize = true;
    args.BetaRange = [0.01,2.8];
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. July 14. (2023a)
%
% 
% `Idx` should be set first

%%

if exist('pansim','var')
    clear pansim
end
clear mex

R = readtimetable(xlsname);
R = pcz_table_mergevars(R);

%%

%%%
% Create simulator object
DIR = fileparts(mfilename('fullpath'));
pansim = ps.mexPanSim_wrap(ps.str2fun([DIR '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
pansim.initSimulation(PanSim_args);

for k = 1:height(R)

    simout = pansim.runForDay(string(R(k,Vn.policy).Variables));
    O = hp.simout2table(simout);
    R(k,O.Properties.VariableNames) = O;

end
clear pansim mex

R = rec_SLPIAHDR(R,R.Date([1,end]),'PWConstBeta',false,'BetaRange',args.BetaRange);

%%

dirname = fullfile("/home/ppolcz/Dropbox/Peti/NagyGep/PanSim_Output",DirName,Name);
if ~exist(dirname,'dir')
    mkdir(dirname)
end

Now = datetime;
Now.Format = "uuuu-MM-dd_HH-mm";
Now = string(Now);

writetimetable(R,fullfile(dirname,Now + ".xls"),"Sheet","Results");
% exportgraphics(fig,fullfile(dirname,Now + ".pdf"),'ContentType','vector');
% exportgraphics(fig,fullfile(dirname,Now + ".jpg"),'ContentType','vector');

end
