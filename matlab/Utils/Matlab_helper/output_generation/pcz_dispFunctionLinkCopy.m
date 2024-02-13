function pcz_dispFunctionLinkCopy(linkname, varargin)
%% pcz_dispFunctionLinkCopy
%  
%  File: pcz_dispFunctionLinkCopy.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 30.
%

%%

if ~G_VERBOSE
    return
end

if ~ischar(linkname)
    linkname = [ inputname(1) ': Copy LaTeX command' ];
end

str = sprintf(varargin{:});

prefix = pcz_dispFunctionGetPrefix;

cmd = sprintf('clipboard(''copy'', [ ''%s'' ])', strrep(str,newline,''' newline '''));

fprintf([ prefix '- '])
pcz_dispHRefMatlab(linkname, cmd)

end