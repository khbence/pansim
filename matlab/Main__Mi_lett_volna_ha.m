%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Modified on 2023. August 3. (2023a)
%
%    THIS SCRIPT SHOULD BE CALLED IN THE ROOT DIRECTORY
% 
%#ok<*CLALL>

clear all

% Population in the simulator
Np = 179500;

%% Load policy measures

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

Sc = "Sc3";

switch Sc
    case "Sc0"
        % Szórakozóhelyeket nem zárjuk be
        % Nincs kijárási korlátozás
        % Iskolákat nem zárjuk be
        % Nincs karantén
        Policy_measures = [ "PLNONE" "CFNONE" "SONONE" "QU0" ];
    case "Sc1"
        % Nincs kijárási korlátozás
        % Iskolákat nem zárjuk be
        Policy_measures = [ "CFNONE" "SONONE" ];
    case "Sc2"
        % Nincs kijárási korlátozás
        Policy_measures = [ "CFNONE" ];
    case "Sc3"
        % Nincs karantén
        Policy_measures = [ "QU0" ];
end
Policy_measures_str = strrep(strjoin(Policy_measures,"_"),".","");

%%

% Load PanSim arguments
PanSim_args = load_PanSim_args("Real");

%%%
% Create simulator object
dir = fileparts(mfilename('fullpath'));
obj = mexPanSim_wrap(str2fun([dir '/mexPanSim'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.initSimulation(PanSim_args);

N = 280;
nout = 49;
simout = zeros(N-1,nout);

for k = 2:N+1
    simout(k-1,:) = obj.runForDay(Policy_measures);
    fprintf('k = %d / %d\n',k-1,N)
end

T = simout2table(simout,1);

fname = "REC_" + string(T.Date(end)) + "_Agent-based_" + Sc + "_" + Policy_measures_str + "_Ujra.mat";
save(fname,"T")

clear all

