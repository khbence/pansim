function [varargout] = create_sym(varname,dim,N,s)
%%
%  File: create_sym.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper/@Pcz_CasADi_Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 29. (2021a)
%

arguments
   
    varname {mustBeValidVariableName}
    dim (1,:) {mustBeNumeric}
    N (1,1) {mustBeNumeric} = 1
    
    s.str {mustBeMember(s.str,[
        "full","sym"
        ])} = "full"
    
    s.CasX {mustBeMember(s.CasX,["SX","MX"])} = "SX"
    
    s.r1 {mustBeMember(s.r1,[
        "Var_full","Var_half","var","f_str","f_vec","defval"
        ])} = "Var_full"
    
    s.r2 {mustBeMember(s.r2,[
        "Var_full","Var_half","var","f_str","f_vec","defval"
        ])} = "Var_half"
    
    s.r3 {mustBeMember(s.r3,[
        "Var_full","Var_half","var","f_str","f_vec","defval"
        ])} = "var"
    
    s.r4 {mustBeMember(s.r4,[
        "Var_full","Var_half","var","f_str","f_vec","defval"
        ])} = "f_str"
    
    s.r5 {mustBeMember(s.r5,[
        "Var_full","Var_half","var","f_str","f_vec","defval"
        ])} = "f_vec"
    
    s.r6 {mustBeMember(s.r6,[
        "Var_full","Var_half","var","f_str","f_vec","defval"
        ])} = "defval"
    
end

import casadi.*

%%%
% Structer-dependent constructions

switch s.str

    case "full" 
        % Dense matrix or cell of dense matrices.
        
        if numel(dim) == 1
            dim(2) = 1;
        end
        
        assert(numel(dim) == 2,join([
            "When declaring a new dense variable," 
            "the third dimension (N) should be given separately."]));
        
        % [1] Number of variables
        n_var = dim(1)*dim(2);
        
        % [2] Sparsity pattern
        sp = Sparsity.dense(dim(1),dim(2));

        % [3] Structure pattern
        structurize_half2full = @(M) {M};
        
        % [4] Default value in a structurized form
        Val = repmat({zeros(dim)},[1,N]);
        
    case "sym" 
        % Symmetric matrix or block diagonal matrix of symmetric blocks or
        % a cell of such a matrix.

        dim_cell = num2cell(dim);

        % dimension of the matrix
        n = sum(dim);
        
        % [1] Number of variables
        n_var = sum(dim .* (dim+1) / 2);

        % [2] Sparsity pattern
        sparsity_cell = cellfun(@(r) {Sparsity.lower(r)},dim_cell);
        sp = diagcat(sparsity_cell{:});
        
        % [3] Structure pattern
        structurize_half2full = @(L) {L + L' - L(Sparsity.diag(n))};
        
        % [4] Default value in a structurized form
        Val = repmat({eye(n)},[1 N]);
end

%%%
% Create CasADi symbolic variables 

if N == 1
    dim_args = {n_var,1};
else
    dim_args = {n_var,1,N};
end

switch s.CasX
    case "SX"
        var_cell = SX.sym(varname,dim_args{:});
        structurize_vec2half = @(var) {SX(sp,var)};
    case "MX"
        var_cell = MX.sym(varname,dim_args{:});
        structurize_vec2half = @(var) {MX(sp,var)};
end

if ~iscell(var_cell)
    var_cell = {var_cell};
end

%%%
% Structer-independent constructions

r = struct;
r.var = vertcat(var_cell{:});
r.Var_half = cellfun(structurize_vec2half,var_cell);
r.Var_full = cellfun(structurize_half2full,r.Var_half);        
r.f_str = Function('structure',{r.var},r.Var_full);
r.f_vec = Function('vectorize',r.Var_half,{r.var});
r.defval = sparse(r.f_vec(Val{:}));

if N == 1
    r.Var_full = r.Var_full{1};
    r.Var_half = r.Var_half{1};
end

%%%
% Select return values

varargout = cell(1,nargout);
for k = 1:nargout
    varargout{k} = r.(s.(sprintf("r%d",k)));
end


end