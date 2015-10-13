clear;
hold off;

% Column description

%    1  -> ID
%    2  -> Width - 2,4,6,8 
%    3  -> ROB size - 32 to 160 
%    4  -> IQ size - 8 to 80
%    5  -> LSQ size - 8 to 80
%    6  -> RF sizes - 40 to 160
%    7  -> RF read ports - 2 to 16
%    8  -> RF write ports - 1 to 8
%    9  -> Gshare size -  1K to 32K
%    10 -> BTB size - 256 to 1024
%    11 -> Branches allowed - 8,16,24,32
%    12 -> L1 Icache size - 64 to 1024
%    13 -> L1 Dcache size - 64 to 1024
%    14 -> L2 Ucache size- 512 to 8K
%    15 -> Depth - 9 to 36
%    16 -> (response) Delay

names = {'Width', 'ROB', 'IQ', 'LSQ', 'RFs', 'RF read', 'RF write', 'Gshare', 'BTB', 'Branches', 'L1 I', 'L1 D', 'L2 U', 'Depth', 'Delay'};
train_input = csvread('../data/train.csv');
valid_input = csvread('../data/validate_and_test.csv');

%training_set = train_input;
%validation = valid_input;
%[training_set, validation] = kfold(train_input, 2);

training_set = train_input(1:600, :);
validation = train_input(601:end, :);

test_set = training_set(:, 2:15);
test_response = training_set(:,end);

lm = fit_cpu(test_set, test_response);

% Validation
valid_id = validation(:,1);
valid_set = validation(:,2:15);
valid_response = validation(:,16);

% Prediction

predict_response = predict(lm, valid_set);
qqplot(valid_response, predict_response)

% Root Mean Squared Error
RMSE = sqrt(mean((valid_response - predict_response).^2));
fprintf('**************************\n');
fprintf('*** RMSE = %f ***\n', RMSE);
fprintf('**************************\n');

%csvwrite('prediction2.csv', [valid_id predict_response]);

%Towards a better model:
% 1. Use feature transformation: 
    %a.By looking at each feature compared to the response variable
    %b.By computing the covariance matrix and study the factors
% 2. Use feature selection:
    %Remove unwanted/useless feature after the transformations
% 3. Build a K-fold Cross Validation algorithm. (K=10 is a good value)
    %Compare the score of each fold (_mean_ + variance) against the
    %test data. If good, run model on test data.

