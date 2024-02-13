function Sh = plot_interval(tt,lb,ub,FaceColor,args)
arguments
    tt (1,:)
    lb (1,:)
    ub (1,:)
    FaceColor = [0 0.4470 0.7410]
    args.LineColor = FaceColor;
    args.LineWidth = 1.2;
    args.LineStyle = '-';
    args.PlotLim = true;
    args.FaceAlpha = 0.2;
end
%%
%  File: plot_interval.m
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2022. June 02. (2022a)
% 

    valid = ~(ismissing(lb) | isinf(lb)) & ~(ismissing(ub) | isinf(ub)) & (lb < ub);


    start1 = circshift([ diff(valid) , 0 ],1) == 1;
    end1 = [ diff(valid) , 0 ] == -1;

    Idx_start = find(start1);
    Idx_end = find(end1);

    if isempty(Idx_start)
        Idx_start = 1;
    end

    if isempty(Idx_end)
        Idx_end = numel(tt);
    end

    if Idx_end(1) < Idx_start(1)
        Idx_start = [1 Idx_start];
    end

    if Idx_end(end) < Idx_start(end)
        Idx_end = [Idx_end numel(tt)];
    end

    assert(numel(Idx_start) == numel(Idx_end), '(BELSO HIBA: plot_interval). Kezdo es vegso valid indexek nincsenek jol kiszamolva.')

    for i = 1:numel(Idx_start)
        Idx = Idx_start(i):Idx_end(i);
    
        Sh = shade(tt(Idx),ub(Idx),tt(Idx),lb(Idx), ...
            'FillType',[1 2;2 1], ...
            'FillColor',FaceColor);
        Sh(3).FaceAlpha = args.FaceAlpha;
    
        if args.PlotLim
            Sh(1).LineStyle = args.LineStyle;
            Sh(2).LineStyle = args.LineStyle;
            Sh(1).LineWidth = args.LineWidth;
            Sh(2).LineWidth = args.LineWidth;
            Sh(1).Color = args.LineColor;
            Sh(2).Color = args.LineColor;
        else
            Sh(1).LineStyle = 'none';
            Sh(2).LineStyle = 'none';
        end
    
    end

end