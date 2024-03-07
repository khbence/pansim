function [ret] = pcz_OK_FAILED(bool, varargin)
%% Script pcz_OK_FAILED
%  
%  file:   pcz_OK_FAILED.m
%  author: Peter Polcz <ppolcz@gmail.com> 
%  
%  Created on 2017.08.01. Tuesday, 14:06:46
%  Modified on 2018.04.04. (április  4, szerda), 20:09
%
%%

if ~islogical(bool) && ~ischar(bool)
    bool = bool == 1;
end

if nargout > 0

    ret = [];

    if islogical(bool)

        if bool
            % fprintf('[  %s  ] ', colorizedstring('green','OK'))

            % 2019.09.06. (szeptember  6, péntek), 17:52
            ret = [ ret sprintf('[   ') ];
            ret = [ ret sprintf('<strong>OK</strong> ') ];
            ret = [ ret sprintf('  ] ') ];
        else
            ret = [ ret sprintf('[ ') ];
            ret = [ ret sprintf('<strong>FAILED</strong> ') ];
            ret = [ ret sprintf('] ') ];
        end

        if ~isempty(varargin)
            ret = [ ret sprintf(varargin{:})];
        end

    elseif ischar(bool)
        ret = [ ret sprintf('[  ') ];
        ret = [ ret sprintf('[\bINFO]\b ') ];
        ret = [ ret sprintf(' ] ') ];
        ret = [ ret sprintf(bool, varargin{:}) ];
    end
    
else

    if islogical(bool)

        if bool
            % fprintf('[  %s  ] ', colorizedstring('green','OK'))

            % 2018.05.21. (május 21, hétfő), 12:35 (ezt kikommentelem)
            fprintf('[   ');
            % cprintf('*green', 'OK ');
            % fprintf('[\b<strong>OK</strong>]\b ');
            fprintf('<strong>OK</strong> ');
            fprintf('  ] ');
        else
            % fprintf('[%s] ', colorizedstring('red','FAILED'))
            fprintf('[ ');
            fprintf(2,'<strong>FAILED</strong> ');
            % cprintf('*err', 'FAILED ');
            fprintf('] ');
        end

        if ~isempty(varargin)
            fprintf(varargin{:})
        end

    elseif ischar(bool)
        fprintf('[  ');
        % fprintf('<a href="">INFO</a> ')
        % fprintf('<strong>INFO</strong> ')
        fprintf('[\bINFO]\b ')
        % cprintf('*blue', 'INFO ');
        fprintf(' ] ');
        fprintf(bool, varargin{:});
    end

end
    
end