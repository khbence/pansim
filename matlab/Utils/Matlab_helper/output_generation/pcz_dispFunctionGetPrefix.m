function [prefix] = pcz_dispFunctionGetPrefix
%% pcz_dispFunctionGetPrefix
%  
%  File: pcz_dispFunctionGetPrefix.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 30.
%

%%

prefix = '';
if G_SCOPE_DEPTH >= 1
    tab = '│   ';
    prefix = repmat(tab,[1 G_SCOPE_DEPTH]);
end


end