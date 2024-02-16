classdef C
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 13. (2023a)
%

properties (Constant = true)
    Np = 179500;
    TrRate_IDX = 1;
    Start_Date = datetime(2020,09,23,"Format","uuuu-MM-dd");

    DIR_PanSim = '/home/ppolcz/Dropbox/Peti/Munka/01_PPKE_2020/PanSim_Results_2';
    DIR_GenLUT = fullfile(C.DIR_PanSim,"GenLUT");
end

end
