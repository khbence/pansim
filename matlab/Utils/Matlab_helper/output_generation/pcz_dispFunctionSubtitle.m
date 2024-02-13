function [ret] = pcz_dispFunctionSubtitle(varargin)
%% pcz_dispFunctionSubtitle
%  
%  File: pcz_dispFunctionSubtitle.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 06.
%


%% make it back-compatible

if nargin > 0 && iscell(varargin{1})
    varargin = varargin{1};
end

%%

if ~G_VERBOSE
    return
end

msg = sprintf(varargin{:});


depth = G_SCOPE_DEPTH;

prefix = '';
if depth >= 1
    tab = '│   ';
    prefix = repmat(tab,[1 depth]);
end

if ~isempty(msg)
    disp([ prefix '' msg])
else
    disp([ prefix ' '])
end

end