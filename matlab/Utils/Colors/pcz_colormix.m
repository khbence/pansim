function [ret] = pcz_colormix(C1,lambda,C2)
arguments
    C1, lambda
    C2 = [1 1 1];
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2023. January 05. (2022b)
%

ret = C1*lambda + C2*(1-lambda);

end