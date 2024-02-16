function s = pcz_mfilename(fname,args)
%%
%  File: pcz_mfilename.m
%  Directory: 5_Sztaki20_Main/Utils/Matlab_helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 26. (2021a)
%
% Use this as follows:
% 
%   pcz_mfilename(mfilename('fullpath'))

arguments
    fname {mustBeTextScalar} = ''
    args.Type {mustBeMember(args.Type,["string","char"])} = "char";
end

if isempty(fname) || contains(fname,'/tmp/Editor')
    fname = matlab.desktop.editor.getActiveFilename;
end

if ~startsWith(fname,filesep)
    error('Valid usage: `s = pcz_mfilename(mfilename(''fullpath''));`')
end

s = struct;
s.version = '2021.05.26. (május 26, szerda), 22:14 [pcz_mfilename]';
s.path = fname;

% 2021.12.07. (december  7, kedd), 11:36 [Kiegeszites]
s.dirs = strsplit(fileparts(fname),filesep);
s.dirs = s.dirs(end:-1:2);

s.dir = fileparts(fname);
s.pdir = fileparts(s.dir);
s.ppdir = fileparts(s.pdir);
s.pppdir = fileparts(s.ppdir);
s.ppppdir = fileparts(s.pppdir);

s.startup = su(s.dir);
s.pstartup = su(s.pdir);
s.ppstartup = su(s.ppdir);
s.pppstartup = su(s.pppdir);
s.ppppstartup = su(s.ppppdir);


[~,s.bname,s.ext] = fileparts(fname);
s.fname = [s.bname s.ext];

s.pdirs = {s.dir s.pdir s.ppdir s.pppdir s.ppppdir};

ret = split(s.dir,'/_/');
if numel(ret) == 2
    s.reldir = ret{2};
else
    s.reldir = '[Relative directory in workspace `_` not detected.]';
end


end

function startup = su(dir)

startup = [dir filesep 'startup.m'];
if ~exist(startup,'file')
    startup = '';
end

end
