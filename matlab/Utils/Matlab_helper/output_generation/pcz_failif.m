function [ret] = pcz_failif(bool, varargin)
%% pcz_failif
%  
%  File: pcz_failif.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. June 09.
%

%%

if ~bool
    pcz_info(bool, varargin{:});
end

if nargout > 0
    ret = ~bool;
end

end