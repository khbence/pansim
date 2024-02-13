function elapsed = pcz_dispFunctionEnd(start_time, depth_method)
%%
%
%  file:   pcz_dispFunctionEnd.m
%  author: Polcz Péter <ppolcz@gmail.com>
%
%  Created on Tue Jan 05 10:50:38 CET 2016
%

%%

elapsed = toc(start_time);

if ~G_VERBOSE
    return
end

if nargin == 1
    depth_method = 1;
end
    
[ST,I] = dbstack;

if depth_method

    depth = G_SCOPE_DEPTH(-1) + 1;

    for i = 2:depth
        fprintf('│   ')
    end
    
    if numel(ST) > I
        disp(['└ ' num2str(elapsed) ' [sec]'])
    end
    
else
    
    for i = I+2:numel(ST)
        fprintf('│   ')
    end
    
    if numel(ST) > I
        disp(['└ ' num2str(elapsed) ' [sec]'])
    end
    
end

evalin('caller', [ 'clear ' inputname(1) ]);

end
