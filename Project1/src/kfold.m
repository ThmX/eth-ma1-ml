function [ T, V ] = kfold( S, k, i )
    [m, ~] = size(S);
    split = ceil(m/k);
    T1 = S(1:(i-1)*split,:);
    T2 = S(i*split+1:end,:);
    T = [T1; T2];
    
    if i ~= k
        V = S((i-1)*split+1:i*split,:);
    else
        V = S((i-1)*split+1:end,:);
    end
end

