function [ret] = pcz_warning(bool, varargin)
%% 
%  
%  file:   pcz_warning.m
%  author: Polcz Péter <ppolcz@gmail.com> 
%  
%  Created on 2017.01.06. Friday, 13:02:42
%

%%

global SCOPE_DEPTH VERBOSE

if nargin == 1 && iscell(bool)
    varargin = bool(2:end);
    bool = bool{1};
end

opts.first = 1;
opts.last = 0;

if nargin > 1 && iscell(varargin{end})
    args = varargin{end};
    varargin = varargin(1:end-1);
    opts = parsepropval(opts,args{:});
end

% S = dbstack;
% S.name

depth = SCOPE_DEPTH;

if VERBOSE

    prefix = '';
    if depth >= 1
        tab = '│   ';
        prefix = repmat(tab,[1 depth]);
    end
    
    % disp([ prefix ' '])

    fprintf(prefix)
    pcz_OK_WARN(bool, varargin{:});
    fprintf('\n')

    % pcz_dispFunctionStackTrace('', 'first', opts.first, 'last', opts.last)
    
    % pcz_dispFunction('Depth = %d', SCOPE_DEPTH)
    
    % disp([prefix '- ' link])
end

if nargout > 0
    if islogical(bool)
        ret = ~bool;
    else
        ret = [];
    end
end
    
% cprintf('_green', 'underlined green');

% web('text://<html><h1>Hello World</h1></html>')


%%

% global SCOPE_DEPTH VERBOSE
% 
% if nargin == 1 && iscell(bool)
%     varargin = bool(2:end);
%     bool = bool{1};
% end
% 
% if isnumeric(bool) && isscalar(bool)
%     bool = bool ~= 0;
% end
%     
% [ST,I] = dbstack;
%     
% depth = SCOPE_DEPTH;
% 
% if VERBOSE
%     
%     if depth >= 0
%         for i = 2:depth
%             fprintf('│   ')
%         end
% 
%         if numel(ST) > I
%             fprintf('│   ')
%         end
%     end
%     
% end
% 
% if islogical(bool)
%        
%     if bool && VERBOSE
%         fprintf('[   ')
%         % cprintf('green', 'OK')
%         % cprintf('text', '  ] ')
%         fprintf('<strong>OK</strong> ');
%         fprintf('  ] ');
%         if ~isempty(varargin), fprintf(varargin{:}); end
%     elseif ~bool
%         fprintf('[  ')
%         fprintf('[\b<strong>WARN</strong>]\b ')
%         % cprintf('*[1,0.5,0]', 'WARN ')
%         fprintf(' ] ')
%         if ~isempty(varargin), fprintf(varargin{:}); end
%     end
% 
% else
% 
%     varargin = [bool varargin];
%     
%     fprintf('[  ')
%     % cprintf('*[1,0.5,0]', 'WARN ')
%     fprintf('[\b<strong>WARN</strong>]\b ')
%     fprintf(' ] ')
%     if ~isempty(varargin), fprintf(varargin{:}); end
%     
% end
% 
% fprintf('\n')
% 
% if nargout > 0
%     if islogical(bool)
%         ret = ~bool;
%     else
%         ret = [];
%     end
% end


end