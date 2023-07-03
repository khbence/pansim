function B = compute_beta(result_avg, population)
    NI = result_avg(23);
    I1 = result_avg(3);
    I2 = 0.75 * result_avg(4);
    I3 = result_avg(5);
    I4 = result_avg(6);
    I5 = 0.1 * result_avg(7);
    I6 = 0.1 * result_avg(8);
    IM = result_avg(27);
    
    B = (NI / (I1 + I2 + I3 + I4 + I5 + I6)) * (population / (population - IM));
end