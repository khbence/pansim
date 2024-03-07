function [ret] = pcz_dispFunction3(varargin)
%% pcz_dispFunction3
%  
%  File: pcz_dispFunction3.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. May 19.
%

%% make it back-compatible

if nargin > 0 && iscell(varargin{1})
    varargin = varargin{1};
end

%%

if ~G_VERBOSE
    return
end

% [ST,I] = dbstack;
% 
% for i = 2:SCOPE_DEPTH
%     fprintf('│   ')
% end

msg = sprintf(varargin{:});

% if numel(ST) > I    
%     if ~isempty(msg)
%         disp(['│   - ' msg])
%     else
%         disp '│   '
%     end
% else
%     disp(['- ' msg ])
% end


depth = G_SCOPE_DEPTH;

prefix = '';
if depth >= 1
    tab = '    ';
    prefix = repmat(tab,[1 depth]);
end

if ~isempty(msg)
    disp([ prefix '' msg])
else
    disp([ prefix ' '])
end

end