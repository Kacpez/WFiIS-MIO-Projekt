function plotActualAndExpectedResultsWithRMSE(fis,x,y)

% Calculate RMSE bewteen actual and expected results
[rmse,actY] = calculateRMSE(fis,x,y);

% Plot results
figure
subplot(1,2,1)
hold on
bar(actY)
bar(y)
bar(min(actY,y),'FaceColor',[0.5 0.5 0.5])
hold off
axis([0 60 0 3.5])
xlabel("Validation input dataset index"),ylabel("Class")
legend(["Actual Class" "Expected Class" "Minimum of actual and expected values"],...
        'Location','NorthWest')
title("RMSE = " + num2str(rmse) + " ")

subplot(1,2,2)
bar(actY-y)
xlabel("Validation input dataset index"),ylabel("Error")
title("Difference Between Actual and Expected Values")

end

