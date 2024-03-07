function [ret] = pcz_dispFunctionStackTrace(varargin)
%% pcz_dispFunctionStackTrace
%  
%  File: pcz_dispFunctionStackTrace.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 06.
%

%%

if ~G_VERBOSE
    return
end

if isempty(varargin) || mod(numel(varargin),2) == 0
    varargin = [ {'Stack trace:'} varargin ];
end

opts.first = 2;
opts.last = 0;
opts = parsepropval(opts,varargin{2:end});

S = dbstack;
% S.name
% opts.first

if ~isempty(varargin{1})
    pcz_dispFunction2(varargin{1})
end

for ii = opts.first+2:numel(S)-opts.last
    link = pcz_dispHRefOpenToLine(S(ii));
    
    if ~strcmp(S(ii).name, 'evaluateCode') && ~contains(S(ii).name, 'subsref') && ~contains(S(ii).name, 'LiveEditorEvaluationHelper') && ~contains(S(ii).name, '@')
        pcz_dispFunction2(['> ' link])    
    end
end

end