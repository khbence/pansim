function [ret] = pcz_dispHRefOpenToLine(arg1, arg2)
%% pcz_dispHRefOpenToLine
%  
%  File: pcz_dispHRefOpenToLine.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 06.
%

%%

if nargin > 1
    file = arg1;
    line = arg2;
    [~,name,~] = fileparts(file);
elseif nargin > 0 && isstruct(arg1)
    file = arg1.file;
    name = arg1.name;
    line = arg1.line;
elseif nargin > 0 && ischar(arg1)
    file = arg1;
    line = 0;
    [~,name,~] = fileparts(file);
end
    
filepath = which(file);

linestr = num2str(line);

if line == 0
    linestr_disp = '';
else
    linestr_disp = [ ':' linestr ];
end

fnscript = pcz_resolvePath(matlab.desktop.editor.getActive().Filename);
if strcmp(name,'LiveEditorEvaluationHelperESectionEval') && exist(fnscript.path,'file')
    % Ha a script-bol Ctrl+Enterrel lett meghivva

    cmd_line = [ 'opentoline(''' fnscript.path ''',' linestr ')' ];
    text = pcz_dispHRefMatlab([ fnscript.bname linestr_disp ], cmd_line);

elseif exist(filepath,'file') && ~strcmp(name,'evaluateCode')

    cmd_line = [ 'opentoline(''' filepath ''',' linestr ')' ];
    text = pcz_dispHRefMatlab([ name linestr_disp ], cmd_line);

else
    
    text = name;
    
    if line > 0
        text = [ text linestr_disp ];
    end
end



if nargout > 0
    ret = text;
elseif G_VERBOSE
    disp(text);
end

end