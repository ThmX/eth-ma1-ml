hold off;

train = csvread('../data/train.csv');
validation = csvread('../data/validate_and_test.csv');

% Column description
%    1  -> Width - 2,4,6,8 
%    2  -> ROB size - 32 to 160 
%    3  -> IQ size - 8 to 80
%    4  -> LSQ size - 8 to 80
%    5  -> RF sizes - 40 to 160
%    6  -> RF read ports - 2 to 16
%    7  -> RF write ports - 1 to 8
%    8  -> Gshare size -  1K to 32K
%    9  -> BTB size - 256 to 1024
%    10 -> Branches allowed - 8,16,24,32
%    11 -> L1 Icache size - 64 to 1024
%    12 -> L1 Dcache size - 64 to 1024
%    13 -> L2 Ucache size- 512 to 8K
%    14 -> Depth - 9 to 36

% Train
delay = train(:,end);
training_set = train(:,2:15);
simple_training_set = train(:,14);

% Model
selected_variable = setdiff(2:15,[3,4,5,8,9,10,11,13]);
mdl_ts = train(:,selected_variable);
lm = LinearModel.fit(mdl_ts, delay)

% Validation
valid_id = validation(:,1);
valid_delay = validation(:,end);
valid_set = validation(:,selected_variable);

% Prediction

predict_delay = predict(lm, valid_set);
qqplot(valid_delay, predict_delay)

csvwrite('prediction.csv', [valid_id predict_delay]);

%Towards a better model:
% 1. Use feature transformation: 
    %a.By looking at each feature compared to the response variable
    %b.By computing the covariance matrix and study the factors
% 2. Use feature selection:
    %Remove unwanted/useless feature after the transformations
% 3. Build a K-fold Cross Validation algorithm. (K=10 is a good value)
    %Compare the score of each fold (_mean_ + variance) against the
    %test data. If good, run model on test data.

