function start_time = pcz_dispFunctionName(subtitle, msg, opts_)
%%
%
%  file:   pcz_dispFunctionName.m
%  author: Polcz Péter <ppolcz@gmail.com>
%
%  Created on Tue Jan 05 10:26:32 CET 2016
%  Modified on 2018. March 30.
%

if nargin > 2 && isstruct(opts_)
    opts_ = pcz_struct2nvpair(opts_);
else
    opts_ = {};
end

opts.parent = 0;
opts = parsepropval(opts,opts_{:});

%% 

if ~G_VERBOSE
    start_time = tic;
    return
end

% Kiegészítve: 2018.03.30. (március 30, péntek), 01:56
if G_SCOPE_DEPTH > 0
    pcz_dispFunction('')
else
    disp(' ')
end

[ST,I] = dbstack;

try
    name = ST(2).name;
    line = ST(2).line;
catch
    name = '';
    line = 0;
end

try
    caller.name = ST(3).name;
    caller.line = ST(3).line;
catch
    caller = '';
end

active = matlab.desktop.editor.getActive;
if pversion >= 2016 && contains(name,'LiveEditor') && ~isempty(active)
    fparts = pcz_resolvePath(active.Filename);
    name = fparts.bname;
end

if nargin > 0 && ~isempty(subtitle)
    if opts.parent && ~isempty(caller) && ~isempty(caller.name)
        name = [ pcz_dispHRefOpenToLine(caller.name, caller.line) ' (' pcz_dispHRefOpenToLine(name) ')' ];
    else
        name = pcz_dispHRefOpenToLine(name,line);
    end
    title = [' - [\b<strong>' subtitle '</strong>]\b'];
else
    if ~isempty(caller) && ~isempty(caller.name)
        name = [ pcz_dispHRefEditFile(name) ' called from ' pcz_dispHRefOpenToLine(caller.name, caller.line) ];
    else
        name =  pcz_dispHRefEditFile(name);
    end
    title = '';
end    

if nargin < 2
    msg = '';
end


depth = G_SCOPE_DEPTH(1);

for i = 2:depth
    fprintf('│   ')
end

msg_text = '';
if ~isempty(msg)
    msg_text = [' [msg:' msg ']'];
end

if numel(ST) > I
    fprintf('%s', ['┌ ' name ])
    fprintf([ title '%s\n' ], msg_text)
end

start_time = tic;


end
