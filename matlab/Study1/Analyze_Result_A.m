%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 11. (2023a)
%
%                                                 ┌─ (R_h)
%                                                 │   ┌─ (R)
%                                                 │   │    ┌─ (D1)
%                                                 │   │    │  ┌─ (D2)
%                                                 │   │    │  │  ┌─ (H)
%                                                 │   │    │  │  │    ┌─ (T)
%                                                 │   │    │  │  │    │    ┌─ (P1)
%                                                 │   │    │  │  │    │    │  ┌─ (P2)
%                                                 │   │    │  │  │    │    │  │   ┌─ (Q)
%                                                 │   │    │  │  │    │    │  │   │  ┌─ (QT)
%                                                 │   │    │  │  │    │    │  │   │  │  ┌─ (NQ)
%                                                 │   │    │  │  │    │    │  │   │  │  │
% 1BasedIdx: 1       2   3    4   5   6   7   8   9   10   11 12 13   14   15 16  17 18 19   [20-21-22-23-24-25] 26     27  28  29   30    31   [32---33---34---35---36---37---38 ] 39  40  [41--42-43-44-45-46-47] 48    49
%            S       E   I1   I2  I3  I4  I5h I6h R_h R    D1 D2 H    T    P1 P2  Q  QT NQ   [ 1    MUT      6 ] HOM    VAC NI  INF  REINF BSTR [ 1  (IMM) immune people?       7 ] HCI HCE [ 1     INFV        7 ] INFH  VNI
% Results:  [178051, 84, 169, 68, 42, 48, 7,  2,  0,  275, 0, 0, 192, 329, 0, 80, 0, 0, 420, [0, 0, 0, 0, 0, 0 ] 12224, 0,  66, 461, 0,    0,   [695, 695, 695, 695, 695, 695, 695] 0,  5,  [66, 0, 0, 0, 0, 0, 0 ] 8,    0]
%                    │   │    │   │   │   │   └─ (H severe) Hospitalized patients                                │      │   │   │    │     │                                        │   └─ (HCE)                    │     └─ (VNI)
%                    │   │    │   │   │   │      with severe symptoms                                            │      │   │   │    │     └─ (BSTR) Booster vaccinated?            └─ (HCI)                        └─ (INFH)
%                    │   │    │   │   │   └─ (H mild) Hospitalized patients with                                 │      │   │   │    └─ (REINF) Reinfected
%                    │   │    │   │   │      mild symptoms                                                       │      │   │   └─ (INF) All infected?
%                    │   │    │   │   └─ (I severe) Infected people in the main                                  │      │   └─ (NI) Newly infected, i.e., new cases
%                    │   │    │   │      sequence of the disease with severe symptoms                            │      └─ (VAC) Vaccinated?
%                    │   │    │   └─ (I mild) Infected people in the main sequence                               └─ (HOM) Home office?
%                    │   │    │      of the disease with mild symptoms
%                    │   │    └─ (A) Asymptomatic people in the main sequence of the
%                    │   │       disease
%                    │   └─ (P) Infected in the presymptomatic phase
%                    └─ (L) Infected in the latent phase, i.e., exposed
% 

R = readtimetable('/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results/Result_2024-02-11_20-17_T1_randref_aggr_Finalized/A.xls');

fig = figure(231);
delete(fig.Children)

Ax = nexttile; hold on, grid on, box on
plot(R.Date,R.NI);

Ax = nexttile; hold on, grid on, box on
plot(R.Date,R.NI);




%%


Idx = 31 + args.ImmuneIdx;

NI = R.NI;   % New cases
L = R.E;     % Latent
P = double(simout(:,3));     % Presymptomatic people
A = double(simout(:,4));     % Asymptomatic people
I1 = double(simout(:,5));    % Infectios people
I2 = double(simout(:,6));    % Infectios people (more severe)
H1 = double(simout(:,7));    % Hospitalized people
H2 = double(simout(:,8));    % Hospitalized people (more severe)
D1 = double(simout(:,11));   % Deceased people
D2 = double(simout(:,12));   % Deceased people (due to more severe symptoms)
IM = double(simout(:,Idx));  % Immune people

% SEIR
S = Np - IM - D1 - D2;
x = [ S L P I1+I2 A ];

Infectious = P + 0.75*A + I1 + I2 + 0.1*(H1+H2);

% Computation
beta = (NI ./ Infectious) .* (Np ./ S);

if ~isscalar(Idx)
    return
end
BETA_MIN = 0.001;
if any(~isfinite(beta)) || any(beta < BETA_MIN)
    warning(['Computed transmission rate is not finite or smaller than %g. \n' ...
        'Cases = %d, infectious = %d, susceptible = %d, immune = %d. \n' ...
        'Transmission rate (beta) = %d. ==> SETTING beta = %g !'], ...
        BETA_MIN,NI,Infectious,S,IM,beta,BETA_MIN);
    beta = BETA_MIN;
end


