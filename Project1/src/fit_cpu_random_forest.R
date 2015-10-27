
train_input <- read.csv('../data/train_r.csv')
valid_input <- read.csv('../data/validate_and_test_r.csv')

# (1 + Width + ROB + IQ + LSQ + RFs + RF.read + RF.write + Gshare + BTB + Branches + L1.I + L1.D + L2.U + Depth)^2

# ROB + IQ + RFs + Gshare + BTB + Branches + L1.I + L1.D + L2.U

set.seed(131)
rf <- randomForest(Delay ~ 1 + Width + ROB + IQ + LSQ + RFs + RF.read + RF.write + Gshare + BTB + Branches + L1.I + L1.D + L2.U + Depth,
                   data = train_input,
                   mtry = 14,
                   ntree=1000)
print(rf)

predicted <- (predict(rf, valid_input))
write.csv(predicted, 'r_predicted7.csv')
