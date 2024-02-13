function R = quantify_policy(R)
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 08. (2023a)
%
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

persistent TP PL CF SO QU MA

if isempty(TP)
    TP = table(categorical(["TPdef"; "TP015"; "TP035"]),[0.5; 1.5; 3.5],'VariableNames',{'TP','TP_Val'});
    PL = table(categorical(["PLNONE";"PL0"]),[0;1],'VariableNames',{'PL','PL_Val'});
    CF = table(categorical(["CFNONE";"CF2000-0500"]),[0;1],'VariableNames',{'CF','CF_Val'});
    SO = table(categorical(["SONONE";"SO3";"SO12"]),[0;1;2],'VariableNames',{'SO','SO_Val'});
    QU = table(categorical(["QU0";"QU2";"QU3"]),[0;2;4],'VariableNames',{'QU','QU_Val'});
    MA = table(categorical(["MA0.8";"MA1.0"]),[0.8;1],'VariableNames',{'MA','MA_Val'});
end

if ~ismember('TP_Val',R.Properties.VariableNames)
    R = join(join(join(join(join(join(R,TP),PL),CF),SO),QU),MA);
end

% round(sqrt([0.5 1.5 3.5] - 0.5)) / 2
R.Iq = [ round(sqrt(R.TP_Val-0.5))/2 , R.PL_Val , R.CF_Val , R.SO_Val/2 , R.QU_Val/4 , (1-R.MA_Val)*5 ];

end