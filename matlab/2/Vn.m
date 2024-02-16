classdef Vn
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 13. (2023a)
%

properties (Constant = true)
    simout = [
...                                              ┌─ (R_h)
...                                              │     ┌─ (R)
...                                              │     │    ┌─ (D1)
...                                              │     │    │    ┌─ (D2)
...                                              │     │    │    │    ┌─ (H)
...                                              │     │    │    │    │    ┌─ (T)
...                                              │     │    │    │    │    │    ┌─ (P1)
...                                              │     │    │    │    │    │    │    ┌─ (P2)
...                                              │     │    │    │    │    │    │    │    ┌─ (Q)
...                                              │     │    │    │    │    │    │    │    │   ┌─ (QT)
...                                              │     │    │    │    │    │    │    │    │   │    ┌─ (NQ)
...                                              │     │    │    │    │    │    │    │    │   │    │
...  1      2    3    4    5    6    7     8     9     10   11   12   13   14   15   16   17  18   19
    "S1"   "E1" "I1" "I2" "I3" "I4" "I5h" "I6h" "R_h" "R1" "D1" "D2" "H_" "T"  "P1" "P2" "Q" "QT" "NQ" ...
...         │    │    │    │    │    │     └─ (H severe) Hospitalized patients
...         │    │    │    │    │    │        with severe symptoms
...         │    │    │    │    │    └─ (H mild) Hospitalized patients with
...         │    │    │    │    │       mild symptoms
...         │    │    │    │    └─ (I severe) Infected people in the main
...         │    │    │    │       sequence of the disease with severe symptoms
...         │    │    │    └─ (I mild) Infected people in the main sequence
...         │    │    │       of the disease with mild symptoms
...         │    │    └─ (A) Asymptomatic people in the main sequence of the
...         │    │       disease
...         │    └─ (P) Infected in the presymptomatic phase
...         └─ (L) Infected in the latent phase, i.e., exposed
...
    "MUT1" "MUT2" "MUT3" "MUT4" "MUT5" "MUT6" ...
...
...  26     27    28   29    30      31    [32     33     34     35     36     37     38  ]  
    "HOM"  "VAC" "NI" "INF" "REINF" "BSTR" "IMM1" "IMM2" "IMM3" "IMM4" "IMM5" "IMM6" "IMM7" ...
...  │      │     │    │     │       │
...  │      │     │    │     │       └─ (BSTR) Booster vaccinated?
...  │      │     │    │     └─ (REINF) Reinfected
...  │      │     │    └─ (INF) All infected?
...  │      │     └─ (NI) Newly infected, i.e., new cases
...  │      └─ (VAC) Vaccinated?
...  └─ (HOM) Home office?
...
...  39    40   [41      42      43      44      45      46      47   ]  48     49
    "HCI" "HCE" "INFV1" "INFV2" "INFV3" "INFV4" "INFV5" "INFV6" "INFV7" "INFH" "VNI"
    ];

    TrRate = "TrRate";
    SLPIA = ["S","L","P","I","A"];
    SLPIAHDR = ["S","L","P","I","A","H","D","R"];

    params = ["tauL","tauP","tauA","tauI","tauH","qA","pI","pH","pD"];

    policy = ["TP","PL","CF","SO","QU","MA"];
    policy_Iq = "Iq";
    policy_Iq_ = Vn.policy_Iq + "_" + string(cellfun(@(i) {num2str(i)},num2cell(1:numel(Vn.policy))))
    TP = table(categorical(["TPdef"; "TP015"; "TP035"]),[0.5; 1.5; 3.5],'VariableNames',{'TP','TP_Val'});
    PL = table(categorical(["PLNONE";"PL0"]),[0;1],'VariableNames',{'PL','PL_Val'});
    CF = table(categorical(["CFNONE";"CF2000-0500"]),[0;1],'VariableNames',{'CF','CF_Val'});
    SO = table(categorical(["SONONE";"SO3";"SO12"]),[0;1;2],'VariableNames',{'SO','SO_Val'});
    QU = table(categorical(["QU0";"QU2";"QU3"]),[0;2;4],'VariableNames',{'QU','QU_Val'});
    MA = table(categorical(["MA0.8";"MA1.0"]),[0.8;1],'VariableNames',{'MA','MA_Val'});
end

methods(Static)
    
    function Iq = Iq(R)
        % round(sqrt([0.5 1.5 3.5] - 0.5)) / 2
        Iq = [ round(sqrt(R.TP_Val-0.5))/2 , R.PL_Val , R.CF_Val , R.SO_Val/2 , R.QU_Val/4 , (1-R.MA_Val)*5 ];
    end
end

end

% TP = [ TP015, TP035, TPdef ]    
% 1.5%-os, 3.5% illetve azt hiszem 0.5% átlagos testingje a népességnek
% 
% PL = [ PL0, PLNONE ]    
% lezárod-e a szórakozó helyeket vagy sem
% 
% CF = [ CF2000-0500, CFNONE ]    
% van e 20-5 curfew
% 
% SO = [ SO12, SO3, SONONE ]    
% 12. osztályig zársz le sulit, 3. osztályig, nincs suli lezárás
% 
% QU = [ QU0, QU2, QU3 ]         
% nincs karantén, te és veled együttlakók bezárása, QU2 + munkahely zárás
% 
% MA = [ MA0.8, MA1.0 ]      
% itt a szám hogy a fertőzés valószínűség mennyivel van maszk által normálva
% 
% TP a tesztelési mennyiség:
% TPdef ami Magyarországon volt úgy kb, 
% TP015, ha a lakosság 1.5%-át teszteljük naponta, 
% TP035 meg ha 3.5%-át
% 
% SO12 hogy csak 12 év alattiak járnak iskolaba, 
% SO3 hogy senki, 
% SONONE hogy mindenki jár
% 
% CF2000-0500 ami volt kijárási korlátozás este 8-reggel 5
% 
% MA0.8 rádob 0.8-as szorzót a fertőzésre kültéri vagy maszkviselős helyszíneken
% 
% QU0 senki nincs karanténozva (diagnosztizált se), 
% QU2 a diagnosztizált és családja, 
% QU3 pedig ha még osztálya, munkatársai is
