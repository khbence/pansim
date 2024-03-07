function T_ = Update_LUT(T,D,dspan)
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 08. (2023a)
%

T_ = T;
D_ = D;

D_(D_.Date <= dspan(1),:) = [];
D_(D_.Date >= dspan(2),:) = [];

Rstr_Vars = policy_varnames;
Dt = join(D_,T_(:,[Rstr_Vars,"Idx","Beta","Intelligent"]),"Keys",Rstr_Vars);

% T_(~T_.Intelligent,:) = [];
% Dt(~Dt.Intelligent,:) = [];

Dg = groupsummary(Dt,"Idx",["mean","std"],"TrRate");

ldx = Dg.GroupCount > 15;
Idx = Dg.Idx(ldx);
T_.Beta(Idx) = Dg.mean_TrRate(ldx);
T_.std_Beta(Idx) = Dg.std_TrRate(ldx);

% T_.Beta = movmean(T_.Beta,5);
% T_.std_Beta = movmean(T_.std_Beta,5);

figure(412);
Tl = tiledlayout(1,1);
ax = nexttile;
hold on, box on, grid on;
% plot(Dt.Idx,Dt.TrRate,'o','MarkerSize',3,'DisplayName','Measurement')
errorbar(Dg.Idx,Dg.mean_TrRate,Dg.std_TrRate,'sk','LineWidth',1,'DisplayName','Measurement mean+-2*std')
stairs([T_.Idx ; T_.Idx(end)+1],T_.Beta([1:end,end]),'LineWidth',2,'DisplayName','LUT');

if false
    
    % B-form spline for the curve
    % sp = spap2(4,4,[T_.Idx([1,1,1]) ; Dt.Idx ; T_.Idx([end,end,end])],[T_.Beta([1,1,1]) ; Dt.TrRate ; T_.Beta([end,end,end])]);
    % stairs(T_.Idx,fnval(sp,T_.Idx),'LineWidth',2,'DisplayName','B-form spline');
    % T_.Beta = fnval(sp,T_.Idx);
    
    % Polynomial fit 
    degree = 1;
    pfit = polyfit(Dt.Idx,Dt.TrRate ./ Dt.Beta,degree);
    T_.Beta = polyval(pfit,T_.Idx) .* T_.Beta;
    
    stairs([T_.Idx ; T_.Idx(end)+1],T_.Beta([1:end,end]),'LineWidth',2,'DisplayName',"polyfit(**," +  num2str(degree) + ")")
    
    
    [beta_min,idx] = min([Dg.mean_TrRate ; T_.Beta(1)]);
    [beta_max,Idx] = max([Dg.mean_TrRate ; T_.Beta(end)]);
        
    
    % Rescale 
    T_.Beta = (T_.Beta - T_.Beta(1)) / (T_.Beta(end) - T_.Beta(1)) * (beta_max - beta_min) + beta_min;
    stairs([T_.Idx ; T_.Idx(end)+1],T_.Beta([1:end,end]),'LineWidth',2,'DisplayName',"polyfit(**," +  num2str(degree) + ") + rescale")

end

YData = (width(T.Iq):-1:0)/60;

[ii,qq] = meshgrid([T.Idx ; T.Idx(end)+1],YData);
Sf = surf(ii,qq,T.Iq([1:end,end],[1:end,end])','HandleVisibility','off');
Sf.FaceAlpha = 0.5;
Sf.EdgeAlpha = 0;
Plot_Colors
colormap(ax,[Color_5;Color_3;Color_2]);

xlim([T.Idx(1),T.Idx(end)+1])

xline(T.Idx(1):10:T.Idx(end),'HandleVisibility','off')

for i = 1:width(T.Iq)
    for Idx = T.Idx(1)+1:30:T.Idx(end)
        Tx = text(Idx,(YData(i)+YData(i+1))/2,Rstr_Vars(i));
    end
end

legend

dspan.Format = 'uuuu-MM-dd';
title(string(dspan(1)) + " -- " + string(dspan(2)))

end