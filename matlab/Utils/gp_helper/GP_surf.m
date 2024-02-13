function [GP_mean,GP_var] = GP_surf(hyp,x1,x2,GP_mean,GP_var)
%%
%  File: GP_plot_1D.m
%  Directory: 5_Sztaki20_Main/Tanulas/11_Exact_Moment_Matching
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. June 18. (2021a)
%

if nargin < 4
    X = [x1(:) x2(:)];
    
    GP_eval(hyp);
    [GP_mean,GP_var] = GP_eval(hyp,X);

    GP_mean = reshape(GP_mean,size(x1));
    GP_std = sqrt(reshape(GP_var,size(x1)));
end

hold on
for alpha = -3:0.1:3
    Sf = surf(x1,x2,GP_mean + alpha*GP_std);
    Sf.CData = GP_std;
    Sf.FaceAlpha = (1 - abs(alpha) / 3)/5;
    shading interp
end

% Plot measurement data
plot3(hyp.X(:,1),hyp.X(:,2),hyp.y,'.','Color','black','MarkerSize',20);

% Colorbar for the standard deviation
colorbar

axis tight

end
