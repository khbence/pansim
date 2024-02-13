function [ret] = pcz_generateString(l, num)
%% 
%  
%  file:   pcz_generateString.m
%  author: Polcz Péter <ppolcz@gmail.com> 
% 
%  Created on 2016.01.17. Sunday, 13:20:32
%

%
%%

if num
    possible_chars = char(['a':'z' '0':'9']);
else 
    possible_chars = char(['a':'z' 'A':'Z']);
end

nr_of_possible_chars = length(possible_chars);

i = ceil(nr_of_possible_chars*rand(1,l));
ret = possible_chars(i); 

end
