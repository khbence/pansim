function [ret] = pcz_dispFunction_v1(argin, depth_method)
%%
%
%  file:   pcz_dispFunction.m
%  author: Polcz Péter <ppolcz@gmail.com>
%
%  Created on 2016.03.04. Friday, 16:59:50
%

if ~G_VERBOSE
    return
end

if nargin < 2
    depth_method = 1;
end

if nargin < 1 
    msg = '';
else
    if ~iscell(argin)
        argin = { argin };
    end
    
    msg = sprintf(argin{:});
end

[ST,I] = dbstack;

if depth_method
    
    depth = G_SCOPE_DEPTH;
    
    for i = 2:depth
        fprintf('│   ')
    end
    
    if numel(ST) > I
        if ~isempty(msg)        
            disp(['│   - ' msg])
        else
            disp '│   '
        end
    end
        
else
    
    for i = I+2:numel(ST)
        fprintf('│   ')
    end
    
    if numel(ST) > I
        disp(['|   - ' msg])
    end
        
end

end