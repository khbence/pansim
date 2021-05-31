%% Inputs
% relative infectiousness: 0.75 for I2; 0.1 for I5_h and I6_h
% accuracyPCR: 1 for E, I...
% accuracyAntigen: 0.25 for E; 0.5 for I1-I4; 0.25 for I5_h and Ih_6

% assume: infected = E, I...

stateNames = {'S','E','I1','I2','I3','I4','I5_h','I6_h','R_h','R','D1','D2'};
WB = {'W','W','W','W','N','M','S','S','S','W','D','D'};
average_days=[-1, 2.5, 3, 4, 4, 4, 10, 10, 14, 365, -1, -1]; %3.2, 2, 3, 3, 3
maximum_days=[-1, 7, 7.5, 13, 13, 13, 20, 20, 21, 730, -1, -1]; %9, 5, 10, 10, 10
ages = [0, 4; 5, 14; 15, 29; 30, 59; 60, 69; 70, 79; 80, -1]; %year
parameters_ages=[
    0.95 0.8 0.7 0.5 0.4 0.3 0.2;
    0.00045 0.00045 0.0042 0.0442 0.1162 0.2682 0.4945;
    0.333 0.333 0.297 0.294 0.292 0.293 0.293;
    0.2 0.2 0.216 0.3 0.582 0.678 0.687]';
% Precondition (probability of death for infected people):
basic = 0.0064;
%illnesses
illnesses=["diabetes",0.0412; "cardiovascular", 0.0637; "chronickidney", 0.0784; "chronicobstructivepulmonary", 0.0862];
%ratio, cube root
illnesses_MultiplierToHealthy = [nthroot(str2double(illnesses(1,2))/basic,3); nthroot(str2double(illnesses(2,2))/basic,3); nthroot(str2double(illnesses(3,2))/basic,3); nthroot(str2double(illnesses(4,2))/basic,3)];
%% Inputs: decreasing the prob. of path towards the fatal outcome directly for healthy agents (and indirectly for agents with preconds)
%parameters_ages(:,4) = parameters_ages(:,4)*0.9; %modifying only the prob. of mortaility
healthy_MultiplierToAverage = nthroot(0.9,3);
%% generate config
fid = fopen(strcat('transition_config.json'),'wt');
solution_config = transition_config(stateNames, WB, ages, illnesses);
str = jsonencode(solution_config);
str = strrep(str, ',', sprintf(',\r'));

str = strrep(str, '{', sprintf('{\r'));
str = strrep(str, '[', sprintf('[\r'));
str = strrep(str, '[{', sprintf('[\r{\r'));

str = strrep(str, '}', sprintf('\r}'));
str = strrep(str, ']', sprintf('\r]'));
str = strrep(str, '}]', sprintf('\r}\r]'));

fprintf(fid, str);
fclose(fid);
%%
for i_age=1:size(parameters_ages,1)
    
    solution_average = transition_average(stateNames, average_days, maximum_days, parameters_ages(i_age,:)); %from Gergely Röst et al.
    %generate healthy:
    for i_states=1:length(solution_average.states)
        if (strcmp(solution_average.states(i_states).stateName, 'I1'))
            solution_healthy = updating_parameters(solution_average, i_states, 'I3', healthy_MultiplierToAverage, true); %I3 and I4
        end
        if (strcmp(solution_average.states(i_states).stateName, 'I3'))
            solution_healthy = updating_parameters(solution_healthy, i_states, 'I6_h', healthy_MultiplierToAverage, false);
        end
        if (strcmp(solution_average.states(i_states).stateName, 'I4'))
            solution_healthy = updating_parameters(solution_healthy, i_states, 'I6_h', healthy_MultiplierToAverage, false);
        end
        if (strcmp(solution_average.states(i_states).stateName, 'I6_h'))
            solution_healthy = updating_parameters(solution_healthy, i_states, 'D1', healthy_MultiplierToAverage, false);
            break
        end
    end
    
    fid = fopen(strcat('transition_illness0_',num2str(i_age),'.json'),'wt');
    str = jsonencode(solution_healthy);
    str = strrep(str, ',', sprintf(',\r'));
    str = strrep(str, '{', sprintf('{\r'));
    str = strrep(str, '[', sprintf('[\r'));
    str = strrep(str, '[{', sprintf('[\r{\r'));
    str = strrep(str, '}', sprintf('\r}'));
    str = strrep(str, ']', sprintf('\r]'));
    str = strrep(str, '}]', sprintf('\r}\r]'));    
    fprintf(fid, str);
    fclose(fid);
    %generate illnesses:
    for i_illness=1:length(illnesses_MultiplierToHealthy)
        for i_states=1:length(solution_healthy.states)
            if (strcmp(solution_healthy.states(i_states).stateName, 'I1'))
                solution = updating_parameters(solution_healthy, i_states, 'I3', illnesses_MultiplierToHealthy(i_illness), true); %I3 and I4
            end
            if (strcmp(solution_healthy.states(i_states).stateName, 'I3'))
                solution = updating_parameters(solution, i_states, 'I6_h', illnesses_MultiplierToHealthy(i_illness), false);
            end
            if (strcmp(solution_healthy.states(i_states).stateName, 'I4'))
                solution = updating_parameters(solution, i_states, 'I6_h', illnesses_MultiplierToHealthy(i_illness), false);
            end
            if (strcmp(solution_healthy.states(i_states).stateName, 'I6_h'))
                solution = updating_parameters(solution, i_states, 'D1', illnesses_MultiplierToHealthy(i_illness), false);
                break
            end
        end
        
        fid = fopen(strcat('transition_illness',num2str(i_illness),'_',num2str(i_age),'.json'),'wt');
        str = jsonencode(solution);
        str = strrep(str, ',', sprintf(',\r'));
        str = strrep(str, '{', sprintf('{\r'));
        str = strrep(str, '[', sprintf('[\r'));
        str = strrep(str, '[{', sprintf('[\r{\r'));
        str = strrep(str, '}', sprintf('\r}'));
        str = strrep(str, ']', sprintf('\r]'));
        str = strrep(str, '}]', sprintf('\r}\r]'));    
        fprintf(fid, str);
        fclose(fid);
    end
end

function solution_new = updating_parameters(solution, from_number, to, multiplier, special)
for i_progressions = 1:length(solution.states(from_number).progressions) %every neighbours
    if (strcmp(solution.states(from_number).progressions{1, i_progressions}.name, to)) %if the neighbour = to
        if (~special)
            remainder_prev = 1 - solution.states(from_number).progressions{1, i_progressions}.chance; %save 1-p
            %modify p:
            if (solution.states(from_number).progressions{1, i_progressions}.chance*multiplier <= 1)
                solution.states(from_number).progressions{1, i_progressions}.chance = solution.states(from_number).progressions{1, i_progressions}.chance*multiplier;
            else
                solution.states(from_number).progressions{1, i_progressions}.chance = 1;
            end
            
            remainder_new = 1 - solution.states(from_number).progressions{1, i_progressions}.chance; %save new 1-p
            for i_progressions_2 = 1:length(solution.states(from_number).progressions) %every neighbours
                if (~strcmp(solution.states(from_number).progressions{1, i_progressions_2}.name, to)) %if the neighbour != to
                    solution.states(from_number).progressions{1, i_progressions_2}.chance = solution.states(from_number).progressions{1, i_progressions_2}.chance/remainder_prev*remainder_new; %modify !p
                end
            end
        else
            remainder_prev = 1 - 2*solution.states(from_number).progressions{1, i_progressions}.chance; %save 1-p
            %modify p:
            if (solution.states(from_number).progressions{1, i_progressions}.chance*multiplier <= 0.5)
                solution.states(from_number).progressions{1, i_progressions}.chance = solution.states(from_number).progressions{1, i_progressions}.chance*multiplier;
                solution.states(from_number).progressions{1, i_progressions+1}.chance = solution.states(from_number).progressions{1, i_progressions+1}.chance*multiplier;
            else
                solution.states(from_number).progressions{1, i_progressions}.chance = 0.5;
                solution.states(from_number).progressions{1, i_progressions+1}.chance = 0.5;
            end
            
            remainder_new = 1 - 2*solution.states(from_number).progressions{1, i_progressions}.chance; %save new 1-p
            for i_progressions_2 = 1:length(solution.states(from_number).progressions) %every neighbours
                if (~strcmp(solution.states(from_number).progressions{1, i_progressions_2}.name, to) && ~strcmp(solution.states(from_number).progressions{1, i_progressions_2}.name, 'I4')) %if the neighbour != to
                    solution.states(from_number).progressions{1, i_progressions_2}.chance = solution.states(from_number).progressions{1, i_progressions_2}.chance/remainder_prev*remainder_new; %modify !p
                end
            end
        end
        break;
    end
end
solution_new = solution;
end

function solution=transition_config(stateName_data, WB_data, ages, illnesses)
    solution.stateInformation.stateNames = stateName_data;
    solution.stateInformation.firstInfectedState = ('E');
    solution.stateInformation.nonCOVIDDeadState = ('D2');
    solution.stateInformation.susceptibleStates(1) = {'S'};
    for i=1:7
        solution.stateInformation.infectedStates(i) = {stateName_data{i+1}};
    end
    
    for i=1:size(ages,1)
        solution.transitionMatrices((i-1)*(size(illnesses,1)+1)+1).fileName = strcat('transition_illness',num2str(0),'_',num2str(i),'.json');
        solution.transitionMatrices((i-1)*(size(illnesses,1)+1)+1).age = [ages(i,1), ages(i,2)];
        solution.transitionMatrices((i-1)*(size(illnesses,1)+1)+1).preCond = num2str(0);
        for j=1:size(illnesses,1)
            solution.transitionMatrices((i-1)*(size(illnesses,1)+1)+j+1).fileName = strcat('transition_illness',num2str(j),'_',num2str(i),'.json');
            solution.transitionMatrices((i-1)*(size(illnesses,1)+1)+j+1).age = [ages(i,1), ages(i,2)];
            solution.transitionMatrices((i-1)*(size(illnesses,1)+1)+j+1).preCond = num2str(j);
        end
    end

    for i=1:length(stateName_data)
        solution.states(i).stateName = stateName_data{i};
        solution.states(i).WB = WB_data{i};
        if (~strcmp(solution.states(i).stateName(1), 'I')  && ~strcmp(solution.states(i).stateName(1), 'E')) %startsWith(solution.states(i).stateName,'I')
            solution.states(i).infectious = 0;
            solution.states(i).accuracyPCR = 0; 
            solution.states(i).accuracyAntigen = 0;
        else
            solution.states(i).accuracyPCR = 1;
            if (strcmp(solution.states(i).stateName(1), 'E'))
                solution.states(i).infectious = 0;
                solution.states(i).accuracyAntigen = 0.25;
            else
                if (endsWith(solution.states(i).stateName,'2'))
                    solution.states(i).infectious = 0.75;
                    solution.states(i).accuracyAntigen = 0.5;
                else
                    if (endsWith(solution.states(i).stateName,'_h'))
                        solution.states(i).infectious = 0.1;
                        solution.states(i).accuracyAntigen = 0.1;
                    else
                        solution.states(i).infectious = 1;
                        solution.states(i).accuracyAntigen = 0.5;
                    end
                end
            end
        end
    end    
end