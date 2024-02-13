function varname = add_sym(o,type,structure,var,s)
%%
%  File: add_sym.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com)
%
%  Created on 2021. April 20. (2020b)
%

arguments
    o

    type {mustBeMember(type,["var","par"])}
    
    % structure of var
    % if s.name is empty, I use the inputname of this argument
    structure
    
    var (:,1) = []

    s.f_str = []
    s.f_vec = []
    s.name {mustBeText} = ''
    s.lb (:,1) = -Inf
    s.ub (:,1) = Inf
    s.val (:,1) {mustBeNumeric} = 0
    s.half = []
end

import casadi.*

%% Sparse arguments

% Find variable's name (if not available)
if isempty(s.name)
    s.name = inputname(3);
end
assert(~isempty(s.name),'The arguments are not correct. I can''t find the variable''s name.')
varname = s.name;

% Make structure a cell (if it is not)
if ~iscell(structure)
    structure = {structure};
end

% Find vector of free variables (if not available)
if isempty(var)
    if ~isempty(s.f_str) && numel(structure) == s.f_str.ni
    
        var = structure;
        structure = s.f_str(var);
    
    elseif ~isempty(s.f_vec)
    
        var = s.f_vec(structure{:});
    
    else  
    
        str_vec = cellfun(@(x) {x(:)},structure);
        var = vertcat(str_vec{:});
        
    end
end

% Create CasADi functions if not provided
if isempty(s.f_str)
    s.f_str = Function('structurize',{var},structure);
end
if isempty(s.f_vec)
    s.f_vec = Function('vectorize',structure,{var});
end

if isempty(s.half)
    s.half = structure;
end

if ~iscell(s.half)
    s.half = { s.half };
end

if numel(s.half) > 1
    s.half = { s.half };
end

%% Value, lower bound, upper bound

numerical_args = ["lb","ub","val"];

main_dim = size(structure{1},1);
n_var = numel(var);

for a = numerical_args

    num = s.(a);
    
    
    assert(numel(num) == n_var || numel(num) == main_dim || isscalar(num),join([...
        "The number of provided numerical values for"
        "'%s' (now=%d) should coincide, the number of"
        "independent parameters (%d)"
        "or it should be a scalar.",...
        ]),a,numel(num),n_var)

    if isscalar(num)
        
        s.(a) = zeros(size(var)) + num;
        
    elseif (numel(num) == main_dim) && (main_dim * round(n_var/main_dim) == n_var)
        
        s.(a) = repmat(num,[round(n_var/main_dim) 1]);
    
    end

end

%% Add parameter/variable to the helper object

switch type
    case "par"
        assert(~isfield(o.P,varname),... || numel(o.P.(varname).par) == n_var,...
        ... 'Parameter `%s` already exists with a different shape. Please construct the helper object again from scratch.',varname)
            'Parameter `%s` already exists in this helper object. Please construct the helper object again from scratch.',varname)

        nx = numel(o.p);
        idx = nx+1:nx+n_var;

        o.P.(varname) = struct(...
            par = var,...
            half = s.half,...
            val = s.val,...
            idx = idx,...
            f_str = s.f_str,...
            f_vec = s.f_vec,...
            set = false);
        
        o.p = [ o.p ; var ];
        o.p_val = [ o.p_val ; s.val ];

    case "var"        
        assert(~isfield(o.X,varname),... || numel(o.X.(varname).var) == n_var,...
        ... 'Variable `%s` already exists with a different shape. Please construct the helper object again from scratch.',varname)
            'Variable `%s` already exists in this helper object. Please construct the helper object again from scratch.',varname)

        nx = numel(o.x);
        idx = nx+1:nx+n_var;

        o.X.(varname) = struct(...
            var = var,...
            half = s.half,...
            lb = s.lb,...
            ub = s.ub,...
            idx = idx,...
            f_str = s.f_str,...
            f_vec = s.f_vec);
        
        o.x = [ o.x ; var ];
        o.lbx = [ o.lbx ; s.lb ];
        o.ubx = [ o.ubx ; s.ub ];
end

end