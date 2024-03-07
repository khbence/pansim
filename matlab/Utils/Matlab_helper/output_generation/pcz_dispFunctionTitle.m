function [ret] = pcz_dispFunctionTitle(varargin)
%% pcz_dispFunctionTitle
%  
%  File: pcz_dispFunctionTitle.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 30.
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
    disp([ prefix ' '])
    disp([ prefix '----------------------------------------------------'])
    disp([ prefix '' msg])
else
    disp([ prefix ' '])
    disp([ prefix '----------------------------------------------------'])
end


end