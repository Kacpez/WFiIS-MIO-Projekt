
clear; clc;
warning('off'); % Input . expects a value in range ., but has a value of .

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
[iris_train, ~, iris_testing] = dividerand(iris_data', P, 0, 1 - P);
[wine_train, ~, wine_testing] = dividerand(wine_data', P, 0, 1 - P);
[seeds_train, ~, seeds_testing] = dividerand(seeds_data', P, 0, 1 - P);

indices_iris = crossvalind('Kfold', size(iris_train, 2), 5);
indices_wine = crossvalind('Kfold', size(wine_train, 2), 5);
indices_seeds = crossvalind('Kfold', size(seeds_train, 2), 5);

iris_training = cell (1, 5);
wine_training = cell (1, 5);
seeds_training = cell (1, 5);

iris_training_x = cell(1, 5);
iris_training_y = cell(1, 5);
iris_testing_x = cell(1, 5);
iris_testing_y = cell(1, 5);

wine_training_x = cell(1, 5);
wine_training_y = cell(1, 5);
wine_testing_x = cell(1, 5);
wine_testing_y = cell(1, 5);

seeds_training_x = cell(1, 5);
seeds_training_y = cell(1, 5);
seeds_testing_x = cell(1, 5);
seeds_testing_y = cell(1, 5);

for i = 1 : 5
    
    iris_training{i} = iris_train(:, find(indices_iris == i));
    wine_training{i} = wine_train(:, find(indices_wine == i));
    seeds_training{i} = seeds_train(:, find(indices_seeds == i));

    iris_training_x{i} = (iris_training{i}(1:4, :))';
    iris_training_y{i} = (iris_training{i}(5, :))';

    wine_training_x{i} = (wine_training{i}(2:14, :))';
    wine_training_y{i} = (wine_training{i}(1, :))';

    seeds_training_x{i} = (seeds_training{i}(1:7, :))';
    seeds_training_y{i} = (seeds_training{i}(8, :))';
    
end

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
DICE_MAX = 12;
ITER_MAX = 100;
POPULATION = 16; % POPULATION > DICE_MAX
NOISE = 0.05;
STALL_MAX = 15;
M_STALL_MAX = 5;

for i = 1 : 1%numel(datasets)
        
    cv5 = 0;
    cv5_tuned = 0;
    
    for fold = 1 : 5
        
        training_set_x = [];
        training_set_y = [];
        for p = 1 : 5
            if p == fold
                continue
            end
            training_set_x = [training_set_x ; datasets{i}{1}{p}];
            training_set_y = [training_set_y ; datasets{i}{2}{p}];
        end
        testing_set_x = datasets{i}{1}{fold};
        testing_set_y = datasets{i}{2}{fold};
    
        % % % Generate FIS
        fis = genfis(training_set_x, training_set_y, options);

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
                fitness_vals(k) = fitness(tested_vals{k}, fis, training_set_x, training_set_y);
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
        
        cv5 = cv5 + sum(round(evalfis(fis, testing_set_x)) == testing_set_y) / numel(datasets{i}{2});
        fis = setTunableValues(fis, getTunableSettings(fis), best_vals);
        cv5_tuned = cv5_tuned + sum(round(evalfis(fis, testing_set_x)) == testing_set_y) / numel(datasets{i}{2});
                      
    end
      
    cv5 = cv5 / 5;
    cv5_tuned = cv5_tuned / 5;
    
end
