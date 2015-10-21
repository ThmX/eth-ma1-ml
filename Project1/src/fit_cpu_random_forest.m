function B = fit_cpu_random_forest( test_set, test_response )
    nTrees = 20;
    B = TreeBagger(nTrees,test_set,test_response, 'Method','regression');
end

