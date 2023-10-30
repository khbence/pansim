%%
%
% 1       2   3   4   5   6   7    8    9   10  11  12  13  14  15  16  17  18  19  20          21      22  23  24  25    26    27                          28  29  30              31    32  (index egybe)
% S       E   I1  I2  I3  I4  I5_h I6_h R_h R   D1  D2  H   T   P1  P2  Q   QT  NQ  MUT         HOM     VAC NI  INF REINF BSTR  IMM                         HCI HCE INFV            INFH  VNI
% 178029  84  183 65  42  46  13   3    0   279 2   0   174 322 2   98  0   0   436 0,0,0,0,0,0 12316   0   70  469 0     0     717,717,717,717,717,717,717 0   10  70,0,0,0,0,0,0  12    0
% 1       2   3   4   5   6   7    8    9   10  11  12  13  14  15  16  17  18  19  20    23    26      27  28  29  30    31    32  33  34  35  36  37  38  39  40  41   43  45  47 48    49  (index kulon)
% 
% 
%       2   3    4    5    6    7     8     9     10  11   12
% IMM = E + I1 + I2 + I3 + I4 + I5h + I6h + R_h + R + D1 + D2   ???
% 
% R_h + R + D1 + D2 = IMM - E - (I1 + I2 + I3 + I4 + I5h + I6h)
% 
% 84 + 169 + 68 + 42 + 48 + 7 +  2 +  0 +  275 + 0 + 0 == 695
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
%
function [x,beta,NI] = get_SEIRb(simout,Np)
%%
    NI = simout(:,28);       % New cases
    I1 = simout(:,3);        % Presymptomatic people
    I2 = 0.75 * simout(:,4); % Asymptomatic people
    I3 = simout(:,5);        % Infectios people
    I4 = simout(:,6);        % Infectios people (more severe)
    I5 = 0.1 * simout(:,7);  % Hospitalized people
    I6 = 0.1 * simout(:,8);  % Hospitalized people (more severe)
    IM = simout(:,38);       % Immune people
    
    % Necessary conversions
    NI = double(NI);
    IM = double(IM);

    % SEIR
    S = Np - IM;
    E = double(simout(:,2)); 
    I = double(I1 + I2 + I3 + I4 + I5 + I6);
    R = IM - E - I;
    x = [ S E I R ];

    % Computation
    beta = (NI ./ I) .* (Np ./ S);
end
