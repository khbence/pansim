function [t] = fg_scale_derivative(t,f,lims,p)
%%
%  File: fg_scale_derivative.m
%  Directory: 5_Sztaki20_Main/Models/01_QArm/v4_2021_05_21/Helper
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2021. May 14. (2021a)
%

arguments
    t (:,1)
    f (:,:)
    lims (:,1)
    p (:,1) = 1
end

[N,dim] = size(f);

errmsg = {'Biztos ezt akarod: size(t) = [%d(N),1], size(f) = [%d(N),%d(dim)], size(lims) = [%d(dim),1], size(p) = [%d(1|dim),1]',...
    size(t,1),N,dim,size(lims,1),size(p,1)};
assert(N == size(t,1),errmsg{:});
assert(dim == size(lims,1),errmsg{:});
assert(isscalar(p) || dim == size(p,1),errmsg{:});
if N <= dim
    error(errmsg{:})
end

if isscalar(p)
    p = ones(dim,1)*p;
end

dt = diff(t);
df = diff(f);

for i = 1:dim

    ub = lims(i) * p(i);
    dt = max(dt,abs(df(:,i))/ub);
    
end

t = cumsum([0;dt]);

end


