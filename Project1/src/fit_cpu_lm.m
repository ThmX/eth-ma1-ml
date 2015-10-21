function lm = fit_cpu( test_set, test_response )
    [~, NVar] = size(test_set);
    
    T = zeros(1,NVar);
    for i = minpts' %1:14 %[1, 4, 6, 7, 14]
        line = zeros(1, NVar);
        line(i) = 1;
        T = [T; line];
    end
    
    lm = fitlm(test_set, test_response, T);
end

