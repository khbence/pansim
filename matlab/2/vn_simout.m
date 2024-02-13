function [ret] = vn_simout
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 08. (2023a)
%

ret = [
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
    "S1"   "E1" "I1" "I2" "I3" "I4" "I5h" "I6h" "R_h" "R1" "D1" "D2"  "H" "T"  "P1" "P2" "Q" "QT" "NQ" ...
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

end