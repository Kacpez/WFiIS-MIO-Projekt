function [rmse,actY] = calculateRMSE(fis,x,y)

persistent evalOptions
if isempty(evalOptions)
    evalOptions = evalfisOptions("EmptyOutputFuzzySetMessage","none", ...
        "NoRuleFiredMessage","none","OutOfRangeInputValueMessage","none");
end

actY = evalfis(fis,x,evalOptions);

del = actY - y;
rmse = sqrt(mean(del.^2));

end