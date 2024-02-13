function [ret] = pcz_num2str_old_v0(...
    A,format,del1,del2,pref, beginning, ending)
%% 
%  
%  file:   pcz_num2str.m
%  author: Polcz Péter <ppolcz@gmail.com> 
%  
%  Created on 2016.03.16. Wednesday, 17:05:51
%

narginchk(1,7);

if nargin < 2 || isempty(format)
    format = '%.4f';
end    

if nargin < 3 || ~ischar(del1)
    del1 = ' , ';
else
    del1 = sprintf(del1);
end

if nargin < 4 || ~ischar(del2)
    del2 = sprintf(' ; ');
else
    del2 = sprintf(del2);
end

if nargin < 5 || ~ischar(pref)
    pref = '';
else
    pref = sprintf(pref);
end

if nargin < 6 || ~ischar(beginning)
    beginning = '';
    if ~isscalar(A)
        beginning = '[ ';
    end
else
    beginning = sprintf(beginning);
end

if nargin < 7 || ~ischar(ending)
    ending = '';
    if ~isscalar(A)
        ending = ' ]';
    end
else
    ending = sprintf(ending);
end

[q,m] = size(A);

delimiters = reshape([repmat({del1}, [m-1,q]) ; repmat({del2}, [1,q])], [1 numel(A)]);

A_char = cellfun(@(a) {sprintf(format,a)}, num2cell(A'));
A_char(1,:) = strcat(repmat({pref}, [1,q]), A_char(1,:));
A_char = reshape(A_char, [1, numel(A)]);

if ~isempty(inputname(1))
    str = [inputname(1) ' = '];
else
    str = '';
end

str = [ str beginning strjoin(A_char, delimiters(1:end-1)) ending ];

if nargout > 0
    ret = str;
else
    disp(str)
end

end