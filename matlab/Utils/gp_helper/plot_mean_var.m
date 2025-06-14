function Sh = plot_mean_var(tt,xx,xx_std,FaceColor,args)
arguments
    tt,xx,xx_std,
    FaceColor = [0 0.4470 0.7410]
    args.LineColor = FaceColor;
    args.LineWidth = 1.2;
    args.LineStyle = '-';
    args.PlotMean = true;
    args.PlotLim = true;
    args.Alpha = 2;
    args.FaceAlpha = 0.2;
end

    Alpha = args.Alpha;
    PlotMean = args.PlotMean;
    args = rmfield(args,"Alpha");
    args = rmfield(args,"PlotMean");

    nv_pairs = repmat(fieldnames(args)',[2,1]);
    for i = 1:width(nv_pairs)
        nv_pairs{2,i} = args.(nv_pairs{2,i});
    end

    if PlotMean
        Pl_Mu = plot(tt,xx,'Color',args.LineColor,'LineStyle',args.LineStyle,"LineWidth",args.LineWidth);
    end
    
    Sh = plot_interval(tt,xx-Alpha*xx_std,xx+Alpha*xx_std,FaceColor,nv_pairs{:});

    if PlotMean
        Sh = [Pl_Mu ; Sh(:)];
    end

end
