function [y] = fitness(x, fis, input, expected_output)
    
    for i = 1 : numel(x)
        if x == 0
            x = 1.0e-7 * rand();  
        end
    end
    
    temp_fis = setTunableValues(fis, getTunableSettings(fis), x);
    y = sum(round(evalfis(temp_fis, input)) == expected_output) / numel(expected_output);
end
