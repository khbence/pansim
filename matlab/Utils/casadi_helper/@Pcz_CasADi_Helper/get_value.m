function var = get_value(o,varname,varargin)
%%
%  File: get_value.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. November 03. (2021b)
%

if ~ischar(varname)
    varname = inputname(2);
end
s = o.X.(varname);

type = 'full';
if numel(varargin) > 1
    type = varargin{1};
end

var = cell(1,s.f_str.n_out);
[var{:}] = s.f_str(o.x_sol(s.idx));

switch type
    case 'full'
        var = cellfun(@(v) {full(v)},var);
    case 'sparse'
        var = cellfun(@(v) {sparse(v)},var);
    case 'DM'
    otherwise
        error('type should be `full|sparse|DM`')
end

if numel(var) == 1
    var = var{1};
end

end
