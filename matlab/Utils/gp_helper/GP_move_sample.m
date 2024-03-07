function [t,X,y,t_,X_,y_] = GP_move_sample(i,t,X,y,t_,X_,y_)
%%
%  File: GP_move_sample.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. January 20. (2020b)
%

% Add ith element to the sparse data set
t_ = [t_ ; t(i)];
X_ = [X_ ; X(i,:)];
y_ = [y_ ; y(i,:)];

% Remove ith element from the dense data set
t = t([1:i-1,i+1:end]);
X = X([1:i-1,i+1:end],:);
y = y([1:i-1,i+1:end],:);

end