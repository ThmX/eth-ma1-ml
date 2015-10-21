function [B, FitInfo] = fit_cpu_lasso( test_set, test_response )
    [B, FitInfo] = lassoglm(test_set, test_response, 'normal', 'Lambda',3:0.1:5, 'CV',10);
end

