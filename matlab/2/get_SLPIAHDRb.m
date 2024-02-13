function [x,beta,NI] = get_SLPIAHDRb(simout,Np,args)
arguments
    simout (:,49)
    Np = 179500
    args.ImmuneIdx = 1;
end
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

%%
    Idx = 31 + args.ImmuneIdx;

    NI = double(simout(:,28));   % New cases
    L = double(simout(:,2));     % Latent
    P = double(simout(:,3));     % Presymptomatic people
    A = double(simout(:,4));     % Asymptomatic people
    I3 = double(simout(:,5));    % Infectios people
    I4 = double(simout(:,6));    % Infectios people (more severe)
    H1 = double(simout(:,7));    % Hospitalized people
    H2 = double(simout(:,8));    % Hospitalized people (more severe)
    D1 = double(simout(:,11));   % Deceased people
    D2 = double(simout(:,12));   % Deceased people (due to more severe symptoms)
    IM = double(simout(:,Idx));  % Immune people
    
    I = I3 + I4;
    H = H1 + H2;
    D = D1 + D2;

    S = Np - L - P - I - A - H - D - IM;
        
    x = [ S L P I A H D IM ];

    Infectious = P + I + 0.75*A + 0.1*H;

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

end
