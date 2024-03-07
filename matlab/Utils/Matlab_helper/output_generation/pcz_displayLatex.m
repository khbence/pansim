function pcz_displayLatex(varargin)
%% 
%  
%  file:   pcz_displatLatex.m
%  author: Polcz Péter <ppolcz@gmail.com> 
%  
%  Created on 2016.02.20. Saturday, 04:15:24
%

if ~G_VERBOSE
    return
end

rules = sed_platex_from_sym;


last_index = nargin;
if nargin > 0 && iscell(varargin{nargin})
    rules = [ rules ; varargin{nargin} ];
    last_index = nargin-1;
end

for k = 1:last_index
    latexname = sed_apply(inputname(k), rules);
    displaystr = pcz_latex(varargin{k}, [], rules);
    
    fprintf('%s = %s\n', latexname, displaystr);
end

end