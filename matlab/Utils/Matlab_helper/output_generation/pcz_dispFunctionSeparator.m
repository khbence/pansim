function [ret] = pcz_dispFunctionSeparator(varargin)
%% pcz_dispFunctionSeparator
%  
%  File: pcz_dispFunctionSeparator.m
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

depth = G_SCOPE_DEPTH;

prefix = '';
if depth >= 1
    tab = '│   ';
    prefix = repmat(tab,[1 depth]);
end

disp([ prefix '----------------------------------------------------'])

end