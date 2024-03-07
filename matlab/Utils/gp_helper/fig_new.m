function fig = fig_new(fignr,name)
%%
%  File: fig_new.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 12. (2021a)
%

arguments
    fignr = -1
    name = ''
end

global Fig_Counter
if isempty(Fig_Counter)
    Fig_Counter = 0;
end
Fig_Counter = Fig_Counter + 1;

if fignr < 1
    fignr = Fig_Counter;
end

fig = figure(fignr);
if ~isempty(name)
    fig.Name = name;
end
delete(fig.Children)

end