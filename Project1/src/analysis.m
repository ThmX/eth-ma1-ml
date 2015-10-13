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
training_set = csvread('../data/train.csv');

% Response variable
delay = training_set(:,end);

% Plot
corrplot(training_set(:,2:16), 'varNames', names);

















% for i = 2:15
%     figure;
%     tested_var = train(:,i);
%     scatter(tested_var, delay);
%     hold on;
%     avg = [];
%     uniques = unique(tested_var)';
%     for u = uniques
%         select = tested_var == u;
%         avg = [avg sum(select .* delay) / sum(select)];
%         hold on;
%     end
%     plot(uniques, avg, 'r');
%     
%     pause;
% end


