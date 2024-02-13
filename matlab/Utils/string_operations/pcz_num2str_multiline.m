function [ret] = pcz_num2str_multiline(varargin)
%% Script pcz_num2str_multiline
%  
%  file:   pcz_num2str_multiline.m
%  author: Peter Polcz <ppolcz@gmail.com> 
%  
%  Created on 2017.07.06. Thursday, 20:37:16
%
%%

first_property = 1;
while first_property <= nargin && isnumeric(varargin{first_property})
    first_property = first_property+1;
end

for i = 1:first_property-1
    str = pcz_num2str(varargin{i}, 'del1', ' ', 'del2', '\n', ...
        'pref', '    ', 'beg', '[\n', ...
        'label', [inputname(i) ' = '], ... '{inputname} = ', ...
        'end', '\n    ];\n', varargin{first_property:end});

    if nargout == 0
        disp(str)
    end
end

if nargout > 0
    ret = str;
end

end