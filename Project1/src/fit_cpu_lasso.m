function [B, FitInfo] = fit_cpu_lasso( test_set, test_response, model )
    LambdaMinDeviance = 5.542230110312592;
    Lambda1SE = 17.526070465354028;
    lambda = LambdaMinDeviance;
    test_pred = x2fx(test_set, model);
    [B, FitInfo] = lassoglm(test_pred, test_response, 'normal', 'Lambda', lambda, 'CV',10);
end

