function [T,VariableNames] = simout2table(simout,args)
arguments
    simout
    args.Population = C.Np;
    args.Date = C.Start_Date
end

N = height(simout);
Day = (0:N-1)';
Date = Day + args.Date;
Date.Format = "uuuu-MM-dd";

[simx,simbeta] = hp.get_SLPIAHDRb(simout,args.Population);

csimout = num2cell(simout,1);
csimx = num2cell(simx,1);

VariableNames = [Vn.TrRate Vn.SLPIAHDR Vn.simout ];

T = timetable(Date,simbeta,csimx{:},csimout{:},'VariableNames',VariableNames);

end
