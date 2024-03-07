function [ret] = pcz_scalar2str(format, scalar)
%% Script pcz_scalar2str
%  
%  file:   pcz_scalar2str.m
%  author: Peter Polcz <ppolcz@gmail.com> 
%  
%  Created on 2017.06.30. Friday, 16:56:22
%
%%

if abs(imag(scalar)) < 1e-10
    ret_ = sprintf(format,scalar);
else
    if sign(imag(scalar)) > 0
        separator = ' + ';
    else 
        separator = ' - ';
    end
    ret_ = sprintf([ format separator format 'i' ], real(scalar), abs(imag(scalar)));
end

if nargout > 0
    ret = ret_;
end

end