
clear; clc;
%warning('off'); % Input . expects a value in range ., but has a value of .

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
        case 'Iris-versicolor'
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

iris_training_x = (iris_training(1:4, :))';
iris_training_y = (iris_training(5, :))';
iris_testing_x = (iris_testing(1:4, :))';
iris_testing_y = (iris_testing(5, :))';

wine_training_x = (wine_training(2:14, :))';
wine_training_y = (wine_training(1, :))';
wine_testing_x = (wine_testing(2:14, :))';
wine_testing_y = (wine_testing(1, :))';

seeds_training_x = (seeds_training(1:7, :))';
seeds_training_y = (seeds_training(8, :))';
seeds_testing_x = (seeds_testing(1:7, :))';
seeds_testing_y = (seeds_testing(8, :))';

% % % Pack datasets to one variable
% {1} - training_x
% {2} - training_y
% {3} - testing_x
% {4} - testing_y
iris_sets = {iris_training_x, iris_training_y, iris_testing_x, iris_testing_y};
wine_sets = {wine_training_x, wine_training_y, wine_testing_x, wine_testing_y};
seeds_sets = {seeds_training_x, seeds_training_y, seeds_testing_x, seeds_testing_y};

datasets = {iris_sets, wine_sets, seeds_sets};
clearvars -except datasets

% FIS Options
options = genfisOptions('SubtractiveClustering');
% DGO Algorithm variables
P = 6; % <- DO NOT MODIFY !!!
DICE_MIN = 1;
DICE_MAX = 6;
ITER_MAX = 100;
POPULATION = 10; % POPULATION > DICE_MAX
NOISE = 0.05;
STALL_MAX = 15;
M_STALL_MAX = 3;

for i = 1 : numel(datasets)
    
    % % % Generate FIS
    fis = genfis(datasets{i}{1}, datasets{i}{2}, options);
        
    % % % Tuning FIS using Dice Game Optimization (DGO)
    % Retrieve FIS parameters
    fis_settings = getTunableSettings(fis);
    param_vals = getTunableValues(fis, fis_settings);
    param_size = size(param_vals);
    lower = min(param_vals);
    upper = max(param_vals);
    
    best_vals = param_vals;
    best_fit = 0;
    previous_mean = 0;
    stall = 0;
    m_stall = 0;   
    fitness_vals = zeros(1, POPULATION);
    tested_vals = cell(1, POPULATION);
    
    for j = 1 : POPULATION
        tested_vals{j} = best_vals + (best_vals * NOISE) .* randn(size(best_vals));
    end
    
    % Main DGO loop
    for j = 1 : ITER_MAX
        
        % 1. Calculate each player`s score
        for k = 1 : POPULATION
            fitness_vals(k) = fitness(tested_vals{k}, fis, datasets{i}{1}, datasets{i}{2});
        end
        score_vals = (fitness_vals - min(fitness_vals)) ./ (sum(fitness_vals) - min(fitness_vals));
               
        % 2. Save best population
        max_index = find(fitness_vals == max(fitness_vals));
        if fitness_vals(max_index) > best_fit
            best_fit = fitness_vals(max_index);
            best_vals = tested_vals{max_index};
            stall = 0;
        else
            stall = stall + 1;
        end
        
        % 3. Each player tosses a dice
        dice_roll = randi([DICE_MIN DICE_MAX], POPULATION);

        % 4. Specify 'guides' for each player
        guides = cell(1, POPULATION);
        for k = 1 : POPULATION
            guides{k} = randperm(POPULATION);
            guides{k} = guides{k}(1 : dice_roll(k));
            while any(guides{k}(:) == k)
                guides{k} = randperm(POPULATION);
                guides{k} = guides{k}(1 : dice_roll(k));
            end
        end

        % 5. Update position of each player
        new_tested_vals = tested_vals;
        for k = 1 : POPULATION
            for l = 1 : numel(guides{k})
                r = sum(rand(1, P), 2) / P;
                new_tested_vals{k} =  new_tested_vals{k} ...
                    + r * (tested_vals{k} - tested_vals{guides{k}(l)}) ...
                    * sign(score_vals(k) - score_vals(guides{k}(l)));
            end
        end
        
        % 6. Rescale values
        for k = 1 : POPULATION
            new_tested_vals{k} = rescale(new_tested_vals{k}, lower, upper);
        end
        tested_vals = new_tested_vals;
        
        % 7. If fitness mean in last M_STALL_MAX iterations has not improved
        % go back to best_vals
        if mean(fitness_vals) > previous_mean
            previous_mean = mean(fitness_vals);
            m_stall = 0;
        elseif m_stall == M_STALL_MAX
            m_stall = 0;
            for k = 1 : POPULATION
                tested_vals{k} = best_vals + (best_vals * NOISE) .* randn(size(best_vals));
            end
        else
            m_stall = m_stall + 1;
        end
        
        % 8. If no improvement in last STALL_MAX iterations - stop
        if stall == STALL_MAX
            break
        end
    end
    
    disp('FIS - learning set');
    disp(sum(round(evalfis(fis, datasets{i}{1})) == datasets{i}{2}) / numel(datasets{i}{2}));
    disp('FIS - testing set');
    disp(sum(round(evalfis(fis, datasets{i}{3})) == datasets{i}{4}) / numel(datasets{i}{4}));
    
    % % % Tune FIS
    fis = setTunableValues(fis, getTunableSettings(fis), best_vals);
    
    disp('tuned FIS - learning set');
    disp(sum(round(evalfis(fis, datasets{i}{1})) == datasets{i}{2}) / numel(datasets{i}{2}));
    disp('tuned FIS - testing set');
    disp(sum(round(evalfis(fis, datasets{i}{3})) == datasets{i}{4}) / numel(datasets{i}{4}));
    
end
