function [ret] = pcz_dispFunctionLatexLink(A, varargin)
%% pcz_dispFunctionLatexLink
%  
%  File: pcz_dispFunctionLatexLink.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 30.
%

%%

if ~G_VERBOSE
    return
end

prefix = pcz_dispFunctionGetPrefix;

label = inputname(1);

str = '';

if ~isnumeric(A) && ~isempty(symvar(A))
    str = pcz_latex_v6(A, 'disp_mode', 2, 'label', [ label ' = '], varargin{:});
    
    if size(A,2) <= 10
        cccccc = repmat('c',[1 size(A,2)]);        
        str = strrep(str, ['\left(\begin{array}{' cccccc '}'], '\pmqty{');
        str = strrep(str, '\end{array}\right)', '}');
    end
    
elseif isnumeric(A)
    str = pcz_num2str_latex([A], varargin{:});
end

cmd = sprintf('clipboard(''copy'', [ ''%s'' ])', strrep(str,newline,''' newline '''));

fprintf([ prefix '- '])
pcz_dispHRefMatlab([ label ': Copy LaTeX command' ], cmd)

end
