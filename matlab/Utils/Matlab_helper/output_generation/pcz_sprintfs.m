function ret = pcz_sprintfs(format, s)
%% 
%  File: pcz_sprintfs.m
%  Directory: utilities/output_generation
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2020. May 14. (2019b)
%
% Example:
%   
%{

  s.arg1 = 123;
  s.arg3 = [123,0.2;1e-5,2];
  s.arg_2__ = 'kutya';
  format = 'Arg1: {arg1,%4.1d}, arg3 = {arg3,%ns}, \nMegint egy kis szoveg: {arg_2__,%s}';
  ret = pcz_sprintfs(format, s)

%}

%%

pattern = '\{(?<name>\w+),(?<format>%[0-9\.\w]+)\}';
match = regexp(format, pattern, 'names');

for i = 1:numel(match)
    formati = match(i).format;
    namei = match(i).name;
    vali = s.(namei);
    
    stri = vali;
    if isnumeric(vali) 
        
        if strcmp(formati,'%ns')
            strijk = cellfun(@(n) {num2str(n)},num2cell(vali));            
            strij = cellfun(@(col) {strjoin(col,',')}, num2cell(strijk,1));
            stri = strjoin( cellfun(@(n) {num2str(n)},strij) , ';' );
        else
            stri = strjoin( cellfun(@(n) {sprintf(formati,n)},num2cell(vali)) , ',' );
        end

        if ~isscalar(vali)
            stri = sprintf('[%s] (%dx%d)', stri, size(vali));
        end
    end
    
    format = regexprep(format, pattern, stri, 'once');
end

ret = strrep(format,'\n',newline);

              
end