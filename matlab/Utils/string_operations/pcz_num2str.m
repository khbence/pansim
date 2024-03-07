function [ret] = pcz_num2str(varargin)
%%
%
%  file:   pcz_num2str.m
%  author: Polcz Péter <ppolcz@gmail.com>
%
%  Created on 2017.02.12. Sunday, 20:49:45
%
% Examples:
% 
%  pcz_num2str(A,B,2*C,A+B,'format','%f', 'del1', ' , ', 'del2', ' ; ',...
%      'pref', '', 'beginning', ' [ ', 'ending', ' ] ', 'round', 5,...
%      'label', '{inputname} = \n', 'name', '2*C')
% 
%  pcz_num2str(A,b,'label','{inputname} = ', 'pref', '    ', 'del2','\n','beg', '[\n')
%
%  A = [
%      8 , 1 , 6
%      3 , 5 , 7
%      4 , 9 , 2 ]
%  b = [
%      0.8147 , 0.6324 , 0.9575 , 0.9572 , 0.4218
%      0.9058 , 0.0975 , 0.9649 , 0.4854 , 0.9157
%      0.127 , 0.2785 , 0.1576 , 0.8003 , 0.7922
%      0.9134 , 0.5469 , 0.9706 , 0.1419 , 0.9595 ]

o.format = '%8.3f';
o.del1 = ' , ';
o.del2 = ' ; ';
o.del2end = '';
o.pref = '';
o.beginning = '[ ';
o.ending = ' ]';
o.label = '';
o.round = 4;
o.name = '[noinputname]';
o.inputname = '{inputname}';

first = 1;
while first <= nargin && isnumeric(varargin{first})
    first = first+1;
end
o = parsepropval(o,varargin{first:end});

nr = first-1;

if nargout > 0
    ret = cell(nr,1);
end

for i = 1:nr
    iname = inputname(i);
    if isempty(iname)
        iname = o.name;
    end

    label = sprintf(o.label);
    if strfind(label, o.inputname)
        label = strrep(label, o.inputname, iname);
    end

    o.var = varargin{i};
    if o.round > 0
        o.var = round(o.var,o.round);
    end

    if isempty(o.var)
        str = '[]';
    else
        str = pcz_num2str_fixed(o.var,o.format, o.del1, o.del2, o.pref, ...
            [label o.beginning], o.ending);
    end
        
    if nargout > 0
        ret{i} = str;
    else
        disp(str)
    end

end

if nargout > 0
    if numel(ret) == 1
        ret = ret{1};
    end
end

end