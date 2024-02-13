function [T] = epid_sigmoid_pattern(ttr,trlen,tspan,args)
arguments
    ttr = datetime(2020,01,01) + [60 120 180]
    trlen = [30 40 50]
    tspan = datetime(2020,01,01) + [0 365]
    args.Names = []
    args.EnableOverlapping = true
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2023. December 19. (2023a)
%

if args.EnableOverlapping
    S = @Interp_Sigmoid_v3;
else
    S = @Interp_Sigmoid_v2;
end

if numel(tspan) == 2 
    dt = 1;
else
    dt = min(diff(tspan));
end

time_range = [ min(tspan(1),ttr(1)-trlen(1)-dt) , max(tspan(end),ttr(end)+trlen(end)+dt) ];
t1 = time_range(1);
t2 = time_range(2);

Date = t1:dt:t2;

m = numel(Date);
n = numel(ttr);

M = zeros(n+1,m);
M(1,:)   = S(t1 , 1 , ttr(1),  trlen(1)   , 0 , t2);
M(end,:) = S(t1 , 0 , ttr(end),trlen(end) , 1 , t2);

for i = 2:n
    M(i,:) = S(t1 , 0 , ttr(i-1),trlen(i-1) , 1 , ttr(i),trlen(i) , 0 , t2);
end

% Normalize the patterns such that their sum is unitary
M = M ./ sum(M,1);

T = {};
T.Date = t1:t2;
T.Pattern = M';

if ~isempty(args.Names)
    if iscell(args.Names)
        args.Names = string(args.Names);
    end
    for i = 1:n 
        T.("V_" + args.Names(i)) = M(i,:)';
    end
end

end

%%

function S = Sigmoid(A,B,N,args)
arguments
    A,B,N
    args.Support = 5;
end
    n = numel(A);
    assert(n == numel(B));

    S = zeros(n,N);
    for i = 1:numel(A)
        a = A(i);
        b = B(i);

        t = linspace(-args.Support,args.Support,N);
        s = -1 ./ (1 + exp(-t));
        
        s = s - s(1);
        s = s / s(end);
        S(i,:) = a + s * (b - a);
    end
end

function [pp,dd] = Interp_Sigmoid_v1(p,Dates)
    Dates = Dates';
    Dates = Dates(:)';

    assert(all(days(diff(Dates)) > 0),'Dates should be given in an increasing order.')

    N = days(Dates(end) - Dates(1)) + 1;

    t = days(Dates - Dates(1)) + 1;
    dd = Dates(1) + (0:N-1);

    pp = zeros(size(p,1),N);

    for i = 1:numel(Dates)-1
        Range = t(i):t(i+1)-1;
        Ni = t(i+1)-t(i);
        if mod(i,2) == 1
            % Zero order hold
            pp(:,Range) = p(:,(i+1)/2) * ones(1,Ni);
        else
            pp(:,Range) = Sigmoid(p(:,i/2),p(:,i/2+1),Ni);
        end
    end
    pp(:,end) = p(:,end);
end

function [pp,dd] = Interp_Sigmoid_v2(d_Start,v,varargin)
    n = nargin/3-1;

    Dates = [varargin{1:3:3*n}]';
    WinR = [varargin{2:3:3*n}]';
    Vals = [v,varargin{3:3:3*n}];

    Dates = [ [d_Start ; Dates + WinR] , [ Dates - WinR ; varargin{end} ] ];

    [pp,dd] = Interp_Sigmoid_v1(Vals,Dates);
end

function [pp,dd] = Interp_Sigmoid_v3(D0,v0,varargin)
    % Number of values to interpolate
    n = (numel(varargin)-1) / 3 + 1;
    Dn = varargin{end};

    dd = D0:Dn;
    pp = zeros(size(dd)) + v0;
    N = numel(pp);
    
    % if only a single value is given, i.e., Interp_Sigmoid_v3(D0,v0,D1)
    if n == 1
        return
    end

    d0 = D0;
    r0 = 0;
    for i = 1:n-1
        d1 = varargin{3*i-2};
        r1 = varargin{3*i-1};
        w1 = 2*r1+1;
        v1 = varargin{3*i};

        if d0+r0 <= d1-r1
            rMid = 1;
            dMid = d0+r0 + ceil(days(d1-r1-d0-r0)/2 - 0.5);
        else
            rMid = ceil(days(d0+r0-d1+r1) / 2 - 0.5);
            dMid = d1-r1 + rMid;
        end

        if i > 1
            alpha = Interp_Sigmoid_v3(D0,0,dMid,rMid,1,Dn);
        else
            alpha = ones(size(pp));
        end

        Transition = Sigmoid(v0,v1,w1);
        Idx_center = days(d1-D0)+1;
        Idx = Idx_center-r1:Idx_center+r1;

        ldx = 1 <= Idx & Idx <= N;

        pp1 = pp;
        pp1(Idx_center:end) = v1;
        pp1(Idx(ldx)) = Transition(ldx);
            
        pp = (1 - alpha) .* pp + alpha .* pp1;

        % plot(dd,pp)
        % grid on

        d0 = d1;
        v0 = v1;
        r0 = r1;
    end

    % plot(dd,pp)
    % grid on
end
