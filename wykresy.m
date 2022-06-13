%% Dane są ustawione na wykonanie wykresów dla zestawu Iris, w celu ich wykonania na reszcie zestawów należy podmienić zestawy danych (datasets)
%Plot the best fitness in each iteration
B=zeros(1,j);
for k=1:j
     B(k)=A(1,k);
end
figure
plot(B(1,:));
xlabel("Iteration")
ylabel("Fitness")
legend("iris","Location","northwest")

%Example of plot the membership functions for an first input variable in the fuzzy inference system fis
figure
plotmf(fis_dice,"input",1)
 
%Evaluate FIS
Out=evalfis(fis_dice, training_set_x);
% Out(Out>0.5)=1;
Out_const=training_set_y;
Out_deafult=evalfis(fis, training_set_x);

%Plot validation data and FIS result
figure
plot(Out_const)
hold on
plot(Out,"o")
hold on
plot(Out_deafult,"x")
hold off
ylabel("Output value")
legend("Validation data","FIS dice","FIS deafult","Location","northwest")
% 
% %Rules
rule=showrule(fis_dice);
% 
% %Generate fuzzy inference system output surface
figure
gensurf(fis_dice)

% Confusion matrix
cm = confusionmat(round(Out_const),round(Out));
figure
confusionchart(cm)

%Plot RMSE
fisout = fis_dice;
fprintf('Training RMSE = %.3f \n',calculateRMSE(fisout,training_set_x,training_set_y));
figure
plotfis(fisout)
%Plot RMSE on testing data
plotActualAndExpectedResultsWithRMSE(fisout,testing_set_x,testing_set_y);


