function [ret] = pcz_OK_WARN(bool, varargin)
%% pcz_OK_WARN
%  
%  File: pcz_OK_WARN.m
%  Directory: 2_demonstrations/lib/matlab
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. May 21.
%

%%


if ~islogical(bool) && ~ischar(bool)
    bool = bool == 1;
end

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
        fprintf('[  ');
        fprintf('[\b<strong>WARN</strong>]\b ')
        % cprintf('*err', 'FAILED ');
        fprintf(' ] ');
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