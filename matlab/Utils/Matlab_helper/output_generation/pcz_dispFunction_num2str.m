function [ret] = pcz_dispFunction_num2str(A, varargin)
%% pcz_dispFunction_num2str
%  
%  File: pcz_dispFunction_num2str.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. April 30.
%  Minor review on 2020. March 17. (2019b)
%

args.format = '%8.3f';
args.del1 = ' ';
args.del2 = ' ; ';
args.del2end = '';
args.pref = '';
args.beginning = '[';
args.ending = ' ]';
args.label = '';
args.round = 4;
args.name = '';
args.inputname = '{inputname}';
args.label = [inputname(1) ' = '];
args = parsepropval(args, varargin{:});

if ~isempty(args.name)
    args.label = [args.name ' = '];
end

%%

if ~G_VERBOSE
    return
end

depth = G_SCOPE_DEPTH;

prefix = '';
if depth >= 1
    tab = '│   ';
    prefix = repmat(tab,[1 depth]);
end

pref2 = repmat(' ',[1 numel(args.label)+1]);

str = [ prefix ...
    pcz_num2str(A, 'name', args.name, 'del1', args.del1, 'del2', [ '\n' prefix pref2 ], ...
    'pref', '    ', 'beg', args.beginning, ...
    'label', args.label, ... '{inputname} = ', ...
    'end', args.ending, varargin{:}) ];

if nargout > 0
    ret = str;
else
    disp([prefix ' '])
    disp(str);
    pcz_dispFunctionStackTrace('', 'first', 1, 'last', 0);
    % disp([prefix ' '])
end

end
