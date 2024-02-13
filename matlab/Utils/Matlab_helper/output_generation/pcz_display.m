function pcz_display(varargin)
%% 
%  
%  file:   pcz_display.m
%  author: Polcz Péter <ppolcz@gmail.com> 
% 
%  Created on Thu Apr 23 13:26:11 CEST 2015
%

%%

if ~G_VERBOSE
    return
end

if nargin == 2 && ischar(varargin{1}) && isempty(inputname(1))
    disp_(varargin{:})
    return
end

for k = 1:nargin
    disp_(inputname(k),varargin{k});
end
    
    function disp_ (name,var)
        if isscalar(var) && isnumeric(var)
            fprintf('%s = %s;\n', name, num2str(var));
            % fprintf('%s [scalar] = %s\n', name, num2str(var));
            return
        elseif isscalar(var) && isa(var,'sym')
            fprintf('%s [sym] = %s\n\n', name, char(var));
            return
        end
        
        s = size(var);
        fprintf([ name ' [' ])
        for i = 1:length(s)-1
            fprintf('%dx',s(i))
        end
        fprintf('%d] = \n',s(end))
        disp ' '
        disp(var)
        disp ' '
    end

end
