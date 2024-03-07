function pcz_dispStruct(s)
%% Script pcz_dispStruct
%  
%  File:   pcz_dispStruct.m
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2017. November 26.
%
%%

if ~G_VERBOSE
    return
end

fns = fieldnames(s);
for i = 1:numel(fns)
    disp([ inputname(1) '.' fns{i} ' = '])
    disp(s.(fns{i}))
end

end