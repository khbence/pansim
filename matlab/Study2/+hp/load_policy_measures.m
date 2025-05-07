function T = load_policy_measures(args)
arguments
    args.filename = ''
    args.savename = ''
    args.reset = false
end

if isempty(args.filename) || isempty(args.savename)
    fp = pcz_mfilename(mfilename('fullpath'));

    Idx = find(strcmp(fp.dirs,'matlab'));
    DIR_Data = [filesep , fullfile(fp.dirs{end:-1:Idx},'Data')];
    
    if isempty(args.filename)
        args.filename = fullfile(DIR_Data,'data2.txt');
    end
    if isempty(args.savename)
        args.savename = fullfile(DIR_Data,'Policy_measures.mat');
    end

    T = hp.load_policy_measures('filename',args.filename,'savename',args.savename);
    T(isnan(T.Pmx),:) = [];
    
    return
end

%%

% Looking for an already computed table
if exist(args.savename,'file') && ~args.reset
    load(args.savename,'T')
    disp("Policy measures loaded from " + args.savename)
    return
end

% Read policy measures, which is a cell array.
matrix = ps.readMatrixFromFile(args.filename);

% The cell array is converted to a matlab table
T = cell2table(matrix,'VariableNames',[Vn.policy,"Week","Beta"]);
T.Week = categorical(T.Week);
Weeks = categories(T.Week);

% Trim variables
for var = Vn.policy
    T.(var) = categorical(strtrim(T.(var)));
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

T = hp.quantify_policy(T);

T = addvars(T,true(height(T),1),'NewVariableNames',"Intelligent",'Before','Beta');

% [Logikai] Ha nincs karantén, akkor nem érdemes intenzívebben tesztelni
T.Intelligent(T.QU_Val == 0 & T.TP_Val > 1,:) = false;

% [Politikai] Ha az iskolat valamilyen mértékben bezárjuk, akkor a szórakozóhelyet is
% zárjuk
T.Intelligent(T.SO_Val > 0 & T.PL_Val == 0,:) = false;

I = T(:,Vn.policy + "_Val");
I.MA_Val = 1 - I.MA_Val;
I = I.Variables;

In = vecnorm(I,1,2);
[~,idx] = min(In);
[~,Idx] = max(In);

% T.Beta(idx) = max( max(T.Beta([1:idx-1 , idx+1:end])) + 0.001, T.Beta(idx) );
% T.Beta(Idx) = min( min(T.Beta([1:Idx-1 , Idx+1:end])) - 0.001, T.Beta(idx) );
T = addvars(T,T.Beta*0.1,'NewVariableNames',"std_Beta",'After',"Beta");

[~,idx] = sort(T.Beta);
T = T(idx,:);

T = addvars(T,(1:height(T))','NewVariableNames',"Idx",'Before',T.Properties.VariableNames(1));
T = addvars(T,T.Idx*NaN,'NewVariableNames',"Pmx",'After',"Idx");
Idx = find(T.Intelligent);
Pmx = 1:numel(Idx);
T.Pmx(Idx) = Pmx;

save(args.savename,'T')

end