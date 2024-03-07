function [ret] = GP_hyp_cell2struct(hyp)
%%
%  File: GP_hyp_cell2struct.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. April 09. (2020b)
%

ret = struct(...
    'ell',hyp{1},...
    'sf',hyp{2},...
    'sn',hyp{3},...
    'X',hyp,{4},...
    'y',hyp,{5});

% ret = { hyp.ell, hyp.sf, hyp.sn, hyp.X, hyp.y };

end