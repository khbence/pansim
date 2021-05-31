function solution = transition_average(stateName_data, average_days, maximum_days, parameters)
    %number of states: 11 --> length of average_days: 11, length of maximum_days: 11
    
    solution.states = states(stateName_data, average_days, maximum_days, parameters);
end

function solution_states = states(stateName_data, average_days, maximum_days, parameters)
    solution_states = struct;
    for i=1:length(stateName_data)
        solution_states(i).stateName = stateName_data{i};
        solution_states(i).avgLength = average_days(i);
        solution_states(i).maxlength = maximum_days(i);
    end
    
    %progressions
    p = parameters(1);
    h = parameters(2);
    kszi = parameters(3);
    mu = parameters(4);
    
    for i=1:length(stateName_data)
        if (strcmp(solution_states(i).stateName, 'S') || strcmp(solution_states(i).stateName, 'R') || strcmp(solution_states(i).stateName(1), 'D'))
            solution_states(i).progressions = {};
        else
            if (strcmp(solution_states(i).stateName, 'E'))
                solution_states(i).progressions = {progression_N_C('I1',1)};
            else
                if (strcmp(solution_states(i).stateName,'I1'))
                    solution_states(i).progressions = {progression_N_C('I2',p), progression_N_C('I3',(1-p)/2), progression_N_C('I4',(1-p)/2)};
                else
                    if (strcmp(solution_states(i).stateName, 'I2') || strcmp(solution_states(i).stateName, 'I5_h') || strcmp(solution_states(i).stateName, 'R_h'))
                        solution_states(i).progressions = {progression_N_C('R',1)};
                    else
                        if (strcmp(solution_states(i).stateName, 'I3') || strcmp(solution_states(i).stateName, 'I4'))
                            solution_states(i).progressions = {progression_N_C('I5_h',h*(1-kszi)), progression_N_C('I6_h',h*kszi), progression_N_C('R',1-h)};
                        else
                            if (strcmp(solution_states(i).stateName, 'I6_h'))
                                solution_states(i).progressions = {progression_N_C('R_h',1-mu), progression_N_C('D1',mu)};
                            end
                        end
                    end
                end
            end
        end
        
        if (~isempty(solution_states(i).progressions))
            for j=1:length(solution_states(i).progressions)
                if (strcmp(solution_states(i).progressions{1,j}.name, 'R') || strcmp(solution_states(i).progressions{1,j}.name, 'R_h'))
                    solution_states(i).progressions{1,j}.isBadProgression = false;
                else
                    solution_states(i).progressions{1,j}.isBadProgression = true;
                end
            end
        end
    end
end

function solution_progression = progression_N_C(name, chance)
    solution_progression.name = name;
    solution_progression.chance = chance;
end