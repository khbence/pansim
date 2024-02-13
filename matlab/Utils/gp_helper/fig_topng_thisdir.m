function [ret] = fig_topng_thisdir(fig,fname)
%%
%  File: fig_topng_thisdir.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 27. (2021a)
%

% fileparts
fp = pcz_mfilename(fname);

actual_directory = pwd;
cd(fp.dir)

figname = strjoin({
    'result/fig'
    num2str(fig.Number)
    '_'
    strrep(fig.Name,' ','_')
    '.png'
    },'');

exportgraphics(fig,figname);

cd(actual_directory)

end
