function [ret] = pcz_fancyDate(varargin)
%% 
%  
%  file:   pcz_fancyDate.m
%  author: Polcz Péter <ppolcz@gmail.com> 
% 
%  Created on Wed Jan 06 17:39:45 CET 2016
%

%%


type = 'comment';
if nargin == 1
    type = varargin{1};
end

if strcmp(type,'comment')
    ret = datestr(now, 'yyyy.mm.dd. dddd, HH:MM:SS');
elseif strcmp(type,'file')
    ret = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
elseif strcmp(type,'var')
    ret = datestr(now, 'HH_MM_SS');
elseif strcmp(type, 'compact')
    ret = datestr(now, 'yymmddHHMMSS');
elseif strcmp(type, 'short')
    ret = datestr(now, 'yyyy.mm.dd.');
elseif strcmp(type, 'informative')
    ret = datestr(now, 'yyyy. mmmm dd.');
end



end
