function [x,beta,NI] = get_SLPIAHDRb_T(R,Np)
arguments
    R
    Np = 179500
end
%%
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
%

%%
    NI = R.NI;         % New cases
    L = R.E1;          % Latent
    P = R.I1;          % Presymptomatic people
    A = R.I2;          % Asymptomatic people
    I = R.I3 + R.I4;   % Infectios people
    H = R.I5h + R.I6h; % Hospitalized people
    D = R.D1 + R.D2;   % Deceased people
    IM = R.IMM1;       % Immune people
    
    S = Np - L - P - I - A - H - D - IM;
        
    x = [ S L P I A H D IM ];

    Infectious = P + I + 0.75*A + 0.1*H;

    % Computation
    beta = (NI ./ Infectious) .* (Np ./ S);

end
