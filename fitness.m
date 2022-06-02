function [y] = fitness(x)
    y = x;
    if (y == 0)
        y = 1.0e-7 * rand();  
    end
    
end

