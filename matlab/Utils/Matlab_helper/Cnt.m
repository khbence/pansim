function ret = Cnt(reset)
%%
%  File: Cnt.m
%  Directory: 4_gyujtemegy/11_CCS/2021_COVID19_analizis/study17_omicron_waning
%  Author: Peter Polcz (ppolcz@gmail.com) 
% 
%  Created on 2022. January 07. (2021b)

persistent k

if nargin > 0
    k = reset;
else
    k = k + 1;
end

ret = k;

end
