function [ret] = pcz_dispHRefMatlab(string, varargin)
%% pcz_dispHRefMatlab
%  
%  File: pcz_dispHRefMatlab.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 06.
%

%%

if nargin == 1
    cmd = string;
elseif nargin == 2
    cmd = varargin{1};
else
    cmd = sprintf(varargin{:});
end

text = [ '<a href="matlab:' cmd '">' string '</a>'];

if nargout > 0
    ret = text;
elseif G_VERBOSE
    disp(text);
end
    
end