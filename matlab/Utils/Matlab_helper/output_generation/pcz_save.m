function [ret] = pcz_save(filename, varargin)
%% 
%  
%  file:   pcz_save.m
%  author: Polcz Péter <ppolcz@gmail.com> 
%  
%  Created on 2016.03.04. Friday, 20:48:09
%

try
    append = 0;
    
    % resolve arguments
    variable_names = cell(1,numel(varargin));
    for i = 1:numel(varargin)
        name = inputname(i+1);
        
        if ~isempty(name)
            % eval(sprintf('%s = varargin{%d};', name, i));
            variable_names{i} = name;
        elseif ischar(varargin{i})
            variable_names{i} = varargin{i};
            
            if strcmp(varargin{i}, '-append'), append = i; end
        end
    end
    
    % [2] resolve filename 
    if ~isempty(inputname(1))
        filename_str = inputname(1);
    else
        filename_str = ['''' filename ''''];
    end
    
    % [3.1] check if append, but file not exists
    if ~exist(filename, 'file') && append
        variable_names{append} = '';
    end
    
    % [4] transform arguments 
    variable_names = cellfun(@(a) {['''' a '''']}, variable_names);
    
    cmd = sprintf('save(%s,%s)', filename_str, strjoin(variable_names, ', '));
    if nargout > 0
        ret = cmd;
    else
        pcz_dispFunction(sprintf('eval: %s', cmd))
        evalin('caller', cmd);
    end
    
catch ex
    getReport(ex)
end


end