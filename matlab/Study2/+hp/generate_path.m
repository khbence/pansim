function Iref = generate_path(Rng_Int,N)

    t_sim = 0:N;

    rng(Rng_Int)
    while true

        Possible_Tks = divisors(N);
        Possible_Tks(Possible_Tks <= 10) = [];
        Possible_Tks(Possible_Tks > 70) = [];
        Tk = Possible_Tks( floor(( rand * (numel(Possible_Tks)-eps) ))+1 );

        Max_Inf = C.Np / 30;

        hyp = {};
        hyp.X = t_sim( sort(randperm(numel(t_sim),N/Tk)) )';
        hyp.y = rand(size(hyp.X)) * Max_Inf;
        hyp.sf = 36;
        hyp.sn = 25;
        hyp.ell = Tk;

        GP_eval(hyp);
        Iref = GP_eval(hyp,t_sim);

        wFnSup = 2.5;
        x = linspace(-wFnSup,wFnSup,numel(t_sim));
        w = normpdf(x,0,1)';
        w = w - w(1);
        [mw,idx] = max(w);
        w = w ./ mw;
        w(idx:end) = 1;

        Iref = Iref .* w;
        if all(Iref >= 0)
            break
        end
    end

end