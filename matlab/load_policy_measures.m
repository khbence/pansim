%%
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

function [T,Rstr_Vars] = load_policy_measures(args)
arguments
    args.filename = 'Data/data2.txt'
    args.savename = 'Data/Policy_measures.mat'
    args.reset = false
end

dir = string(fileparts(mfilename("fullpath"))) + filesep;

% Restriction abbreviations
Rstr_Vars = ["TP","PL","CF","SO","QU","MA"];

% Looking for an already computed table
if exist(dir + args.savename,'file') && ~args.reset
    load(dir + args.savename,'T')
    disp("Policy measures loaded from " + args.savename)
    return
end

% Read policy measures, which is a cell array.
matrix = readMatrixFromFile(dir + args.filename);

% The cell array is converted to a matlab table
T = cell2table(matrix,'VariableNames',[Rstr_Vars,"Week","Beta"]);
T.Week = categorical(T.Week);
Weeks = categories(T.Week);

% Trim variables
for col = Rstr_Vars
    T.(col) = strtrim(T.(col));
end


%{
T =    TP            PL              CF               SO          QU          MA        Week      Beta  
    _________    __________    _______________    __________    _______    _________    _____    _______
    {'TPdef'}    {'PLNONE'}    {'CFNONE'     }    {'SONONE'}    {'QU0'}    {'MA1.0'}    WEEK0    0.22251
    {'TPdef'}    {'PLNONE'}    {'CFNONE'     }    {'SONONE'}    {'QU0'}    {'MA1.0'}    WEEK1    0.20982
        :            :                :               :            :           :          :         :   
    {'TP035'}    {'PL0'   }    {'CF2000-0500'}    {'SO3'   }    {'QU3'}    {'MA0.8'}    WEEK8    0.13895
	Contains 1944 rows.
%}

T = unstack(T,"Beta","Week");

%{
T =    TP            PL              CF               SO          QU          MA         WEEK0      WEEK1      WEEK2      WEEK3      WEEK4      WEEK5      WEEK6      WEEK7      WEEK8 
    _________    __________    _______________    __________    _______    _________    _______    _______    _______    _______    _______    _______    _______    _______    _______
    {'TPdef'}    {'PLNONE'}    {'CFNONE'     }    {'SONONE'}    {'QU0'}    {'MA1.0'}    0.22251    0.20982    0.21248    0.21439    0.21382    0.21198    0.21257    0.21202    0.21315
    {'TPdef'}    {'PLNONE'}    {'CFNONE'     }    {'SONONE'}    {'QU0'}    {'MA0.8'}     0.2042    0.18848    0.18983     0.1918    0.18907    0.18943    0.18983    0.18958    0.19056
        :            :                :               :            :           :           :          :          :          :          :          :          :          :          :   
    {'TP035'}    {'PL0'   }    {'CF2000-0500'}    {'SO3'   }    {'QU3'}    {'MA0.8'}    0.17327    0.13887    0.14006    0.14132    0.14049    0.13933    0.13924    0.13898    0.13895
	Contains 216 rows.
%}

% Compute mean beta obtained for the different weeks
Beta_mean = mean( T(:,Weeks).Variables , 2 );

% Alter the table such that only the mean beta is 
T = addvars(T,Beta_mean,'NewVariableNames',"Beta",'Before',Weeks{1});

%{
T      TP            PL              CF               SO          QU          MA         Beta       WEEK0      WEEK1      WEEK2      WEEK3      WEEK4      WEEK5      WEEK6      WEEK7      WEEK8 
    _________    __________    _______________    __________    _______    _________    _______    _______    _______    _______    _______    _______    _______    _______    _______    _______
    {'TPdef'}    {'PLNONE'}    {'CFNONE'     }    {'SONONE'}    {'QU0'}    {'MA1.0'}    0.21364    0.22251    0.20982    0.21248    0.21439    0.21382    0.21198    0.21257    0.21202    0.21315
    {'TPdef'}    {'PLNONE'}    {'CFNONE'     }    {'SONONE'}    {'QU0'}    {'MA0.8'}    0.19142     0.2042    0.18848    0.18983     0.1918    0.18907    0.18943    0.18983    0.18958    0.19056
        :            :                :               :            :           :           :          :          :          :          :          :          :          :          :          :   
    {'TP035'}    {'PL0'   }    {'CF2000-0500'}    {'SO3'   }    {'QU3'}    {'MA0.8'}    0.14339    0.17327    0.13887    0.14006    0.14132    0.14049    0.13933    0.13924    0.13898    0.13895
	Contains 216 rows.
%}

save(dir + args.savename,'T')

end