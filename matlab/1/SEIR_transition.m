function dx = SEIR_transition(x, u)

S = x(1);
E = x(2);
I = x(3);

dt = 0.02;
% k2 = 0.37; % TODO
% k3 = 0.1429; % TODO

k2 = u(2);
k3 = u(3);

dS = -u(1)*S*I;
dE = +u(1)*S*I - k2*E;
dI = k2*E - k3*I;
dR = k3*I;

dx = x +dt*[dS; dE; dI; dR];