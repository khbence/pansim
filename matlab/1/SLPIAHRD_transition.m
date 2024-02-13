function dx = SLPIAHRD_transition(x, u)
p = [1/2.5; 1/3; 1/3; 0.75; 0.6; 1/4; 1/4; 0.076; 1/10; 0.145];
N_population = 1;
dt = 0.02;

f = [-u*(x(3) + x(4) + p(4)*x(5))*x(1)/N_population;
     +u*(x(3) + x(4) + p(4)*x(5))*x(1)/N_population - p(1)*x(2);
     p(1)*x(2) - p(2)*x(3);
     p(5)*p(2)*x(3) - p(6)*x(4);
     (1-p(5))*p(2)*x(3) - p(7)*x(5);
     p(6)*p(8)*x(4) - p(9)*x(6);
     p(6)*(1-p(8))*x(4) + p(7)*x(5) + (1-p(10))*p(9)*x(6);
     p(10)*p(9)*x(6)
];

dx = x +dt*f;