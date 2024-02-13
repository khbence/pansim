function T = epid_get_variant_dominance_pattern(Q,time_range,args)
arguments
    Q, time_range
    args.EnableOverlapping = true
    args.TrLen = 'TransitionLength'
end
%%
%  File: epid_get_variant_dominance_pattern.m
%  Directory: /home/ppolcz/T/_Epid/Utils/epid_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2022. July 11. (2022a)
%  Revised on 2022. September 09. (2022a)
%  Revised on 2022. September 12. (2022a)
%

if args.EnableOverlapping
    % 2022.09.09. (szeptember  9, péntek), 13:25
    S = @Epid_Par.Interp_Sigmoid_v3;
else
    S = @Epid_Par.Interp_Sigmoid_v2;
end

if nargin == 1
    time_range = [Q.Date(1) Q.Date(end) + Q.(args.TrLen)(end) + 1];
end

if numel(time_range) > 2
    time_range = time_range([1,end]);
end

% Ensure that the starting date and the ending date are appropriate for
% sigmoid interpolation
Start_Date = min(time_range(1),Q.Date(1));
End_Date = max(time_range(2),Q.Date(end) + Q.(args.TrLen)(end) + 1);

Date = (Start_Date:End_Date)';
T = timetable(Date);

m = numel(Date);
n = height(Q);

M = zeros(n,m);
M(1,:)   = S(Start_Date , 1 , Q.Date(2),  Q.(args.TrLen)(2)   , 0 , End_Date);
M(end,:) = S(Start_Date , 0 , Q.Date(end),Q.(args.TrLen)(end) , 1 , End_Date);

for i = 2:n-1
    M(i,:) = S(Start_Date , 0 , Q.Date(i),Q.(args.TrLen)(i) , 1 , Q.Date(i+1),Q.(args.TrLen)(i+1) , 0 , End_Date);
end

% Normalize the patterns such that their sum is unitary
M = M ./ sum(M,1);

T.Pattern = M';

for i = 1:n
    T.("V_" + Q.Properties.RowNames{i}) = M(i,:)';
end

% Chop data if the required time range is smaller
T = T(time_range(1) <= T.Date & T.Date <= time_range(2),:);

end