function [Pl_sample,Sh,GP_mean,GP_var] = GP_plot_1D(hyp,x,GP_mean,GP_var)
%%
%  File: GP_plot_1D.m
%  Directory: 5_Sztaki20_Main/Tanulas/11_Exact_Moment_Matching
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. June 18. (2021a)
%

if nargin < 4
    [GP_mean,GP_var] = GP_eval(hyp,x);
end

Color_Shade = [ 0.85 , 0.325 , 0.098 ];
Color_Sample = [ 0 , 0 , 0 ];

GP_std = sqrt(GP_var);

Pl_sample = plot(hyp.X,hyp.y,'.','Color',Color_Sample);
hold on

xlabel('$x$','Interpreter','latex','FontSize',14)
ylabel('$f(x)$','Interpreter','latex','FontSize',14)

Logger.latexify_axis(gca,12);

Sh = plot_mean_var(x,GP_mean,GP_std,Color_Shade);

grid on

end
