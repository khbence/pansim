function T = param2table(Param)
arguments
    Param
end

K = Epid_Par.GetK;

idx = zeros(1,numel(Vn.params));
i = 1;
for vn = Vn.params
    idx(i) = K.(vn);
    i = i+1;
end

T = array2table(Param(:,idx),'VariableNames',Vn.params);
