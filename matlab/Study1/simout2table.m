function [T,simx,simbeta] = simout2table(simout,Start_Day,args)
arguments
    simout
    Start_Day = 0
    args.Start_Date = datetime(2020,09,23) + Start_Day
    args.Population = 179500
    args.Getter = @get_SEIRb
    args.VariableNames = ["S" "E" "I" "R"]
end

N = height(simout);
Day = (0:N-1)' + Start_Day;
Date = Day + args.Start_Date;
Date.Format = "uuuu-MM-dd";

[simx,simbeta] = args.Getter(simout,args.Population);

csimout = num2cell(simout,1);
csimx = num2cell(simx,1);

VariableNames = ["TrRate" args.VariableNames simout_varnames ];

T = timetable(Date,simbeta,csimx{:},csimout{:},'VariableNames',VariableNames);

end
