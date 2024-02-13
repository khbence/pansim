function [ret] = pcz_dispHRefEditFile(filename)
%% pcz_dispHRefEditFile
%  
%  File: pcz_dispHRefEditFile.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 06.
%

%%

text = ['<a href="matlab:edit(''' which(filename) ''')">' filename '</a>'];

if nargout > 0
    ret = text;
elseif G_VERBOSE
    disp(text);
end


end