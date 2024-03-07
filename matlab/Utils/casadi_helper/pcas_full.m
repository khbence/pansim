function [varargout] = pcas_full(F,varargin)
%%
%  File: pcas_full.m
%  Directory: 5_Sztaki20_Main/Utils/casadi_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. November 03. (2021b)
%

ret = cell(1,nargout);

[ret{:}] = F(varargin{:});

varargout = cellfun(@full,ret,"UniformOutput",false);

end