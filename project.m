
% % % Load UCI datasets

% % IRIS
% 1. sepal length in cm
% 2. sepal width in cm
% 3. petal length in cm
% 4. petal width in cm
% 5. class:
%  1 - Iris-setosa
%  2 - Iris-versicolour
%  3 - Iris-virginica
temp = importdata('iris.data');
iris_data = zeros(length(temp), 5);
for i = 1 : length(temp)
    row = split(temp(i, 1), ',');
    iris_data(i, 1:4) = str2double((row(1:4)).');
    switch row{5}
        case 'Iris-setosa'
            iris_data(i, 5) = 1;
        case 'Iris-versicolour'
            iris_data(i, 5) = 2;
        case 'Iris-virginica'
            iris_data(i, 5) = 3;
    end
end

% % WINE
% 1. Class: 1, 2, 3
% 2. Alcohol
% 3. Malic acid
% 4. Ash
% 5. Alcalinity of ash
% 6. Magnesium
% 7. Total phenols
% 8. Flavanoids
% 9. Nonflavanoid phenols
% 10. Proanthocyanins
% 11. Color intensity
% 12. Hue
% 13. OD280/OD315 of diluted wines
% 14. Proline 
wine_data = importdata('wine.data');

% % SEEDS
% 1. area A,
% 2. perimeter P,
% 3. compactness C = 4*pi*A/P^2,
% 4. length of kernel,
% 5. width of kernel,
% 6. asymmetry coefficient
% 7. length of kernel groove.
% 8. class: 1, 2, 3
seeds_data = importdata('seeds_dataset.txt');
% remove rows with NaN
seeds_data(any(isnan(seeds_data), 2), :) = []; 

% % % Divide data to training (70%) and testing (30%) sets 
P = 0.7;
[iris_training, ~, iris_testing] = dividerand(iris_data', P, 0, 1 - P);
[wine_training, ~, wine_testing] = dividerand(wine_data', P, 0, 1 - P);
[seeds_training, ~, seeds_testing] = dividerand(seeds_data', P, 0, 1 - P);
 
clearvars -except iris_training iris_testing wine_training wine_testing seeds_training seeds_testing


