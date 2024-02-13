function [report] = pcz_getReport(ex)
%% pcz_getReport
%  
%  File: pcz_getReport.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. May 22.
%

%%


report = getReport(ex);
report = strrep(report, newline, [ newline pcz_dispFunctionGetPrefix ]);

end