clear;

train_input = csvread('../data/train.csv');
valid_input = csvread('../data/validate_and_test.csv');

training_set = train_input;
validation = valid_input;

% Data processing
test_set_raw = training_set(:, 2:8);
test_set_conv = conv2(max(log(test_set_raw), -20), fspecial('gaussian', [5 1], .75), 'same');
test_set = test_set_norm;
test_response = training_set(:,end);

% Validation
valid_id = validation(:,1);
valid_set_raw = validation(:,2:8);
valid_set_conv = conv2(max(log(valid_set_raw), -20), fspecial('gaussian', [5 1], .75), 'same');
valid_set = valid_set_norm;

% Fitting
t = templateSVM('Standardize',1,'KernelFunction','gaussian');
mdl = fitcecoc(test_set,test_response,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'0','1','2'},...
    'Verbose',2);

% Prediction
[predict_response, score] = predict(mdl, valid_set);

% Writing to file

res = horzcat(num2cell(valid_id), predict_response);
table = array2table(res, 'VariableNames', {'Id', 'Label'});
writetable(table , 'prediction4.csv');

