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

model = 'quadratic';
RMSEs = [];
k = 6;
for i = 1:k
    [training_set, validation] = kfold(train_input, k, i);

    test_set = training_set(:, 2:15);
    test_response = training_set(:,end);

    % Validation
    valid_set = x2fx(validation(:,2:15), model);
    valid_response = validation(:,16);
    
    % Fitting
    [B, FitInfo] = fit_cpu_lasso(test_set, test_response, model);

    % Prediction
    predict_response = valid_set * B(:,FitInfo.IndexMinDeviance) + FitInfo.Intercept(FitInfo.IndexMinDeviance);

    % Root Mean Squared Error
    rmse = sqrt(mean((valid_response - predict_response).^2));
    
    scatter(i, rmse, 'b');
    hold on;
    RMSEs = [RMSEs rmse];
end

hold on;
RMSE_mean = mean(RMSEs);
RMSE_sd = std(RMSEs);
plot(1:k, repmat(RMSE_mean,1,k), 'r');
plot(1:k, repmat(RMSE_mean+RMSE_sd,1,k), 'g');
plot(1:k, repmat(RMSE_mean-RMSE_sd,1,k), 'g');

fprintf('***************************\n');
fprintf('***   mean = %f ***\n', RMSE_mean);
fprintf('***   sd   = %f ***\n', RMSE_sd);
fprintf('***************************\n');
