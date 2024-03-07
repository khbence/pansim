function [ret] = pcz_num2str_fixed(A,format,del1,del2,pref,beginning,ending,del2end)
%% 
%  
%  file:   pcz_num2str.m
%  author: Polcz Péter <ppolcz@gmail.com> 
%  
%  Created on 2016.03.16. Wednesday, 17:05:51
%

narginchk(7,7);

del1 = sprintf(del1);
del2 = sprintf(del2);
pref = sprintf(pref);
beginning = sprintf(beginning);
ending = sprintf(ending);

[q,m] = size(A);

delimiters = reshape([repmat({del1}, [m-1,q]) ; repmat({del2}, [1,q])], [1 numel(A)]);

A_char = cellfun(@(a) {pcz_scalar2str(format,a)}, num2cell(A'));
A_char(1,:) = strcat(repmat({pref}, [1,q]), A_char(1,:));
A_char = reshape(A_char, [1, numel(A)]);

if ~isempty(inputname(1))
    str = [inputname(1) ' = '];
else
    str = '';
end

str = [ str beginning strjoin(A_char, delimiters(1:end-1)) , ending ];

if nargout > 0
    ret = str;
else
    disp(str)
end

end