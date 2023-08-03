function [x,beta,NI] = get_SEIRb(simout,Np)
%%
    NI = simout(28);       % New cases
    I1 = simout(3);        % Presymptomatic people
    I2 = 0.75 * simout(4); % Asymptomatic people
    I3 = simout(5);        % Infectios people
    I4 = simout(6);        % Infectios people (more severe)
    I5 = 0.1 * simout(7);  % Hospitalized people
    I6 = 0.1 * simout(8);  % Hospitalized people (more severe)
    IM = simout(38);       % Immune people
    
    % Necessary conversions
    NI = double(NI);
    IM = double(IM);

    % SEIR
    S = Np - IM;
    E = double(simout(2)); 
    I = double(I1 + I2 + I3 + I4 + I5 + I6);
    R = IM - E - I;
    x = [ S ; E ; I ; R ];

    % Computation
    beta = (NI / I) * (Np / S);
end