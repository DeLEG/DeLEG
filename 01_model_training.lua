require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

--totalNumLines
data_fer_file = io.open('1.txt')
numLines = 0
for line in data_fer_file:lines() do
    numLines = numLines+1
end
print(numLines)
data_fer = torch.Tensor(numLines,6000)
data_fer_labels = torch.ones(numLines)
data_fer_file = io.open('1.txt')
LineNum = 0;
for line in data_fer_file:lines() do
    LineNum=LineNum+1
    A = line:split(" ")
    if (#A) == 6000 then
        data_fer[LineNum]=torch.Tensor(A)
    end
end

data_infer_file = io.open('0.txt')
numLines = 0
for line in data_infer_file:lines() do
numLines = numLines+1
end
print(numLines)
data_infer = torch.Tensor(numLines,6000)
data_infer_labels = torch.ones(numLines)*2
data_infer_file = io.open('0.txt')
LineNum = 0;
for line in data_infer_file:lines() do
    LineNum=LineNum+1
    A = line:split(" ")
    if (#A) == 6000 then
        data_infer[LineNum]=torch.Tensor(A)
    end
end

print(#data_fer)
print(#data_infer)

trainData = torch.cat({data_fer,data_infer},1)
trainLabels = torch.cat({data_fer_labels,data_infer_labels},1)
print(#trainData)
print(#trainLabels)

mean = trainData:mean()
std = trainData:std()
trainData:add(-mean)
trainData:div(std)
print(trainData:mean())
print(trainData:std())
trainData = trainData:reshape(trainData:size(1),trainData:size(2),1)

model = require 'model.lua'

N = trainData:size(1)

local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion()

local x,y

local feval = function(params)
if theta~=params then
theta:copy(params)
end
gradTheta:zero()
out = model:forward(x)
--print(#x,#out,#y)
local loss = criterion:forward(out,y)
local gradLoss = criterion:backward(out,y)
model:backward(x,gradLoss)
return loss, gradTheta
end

batchSize = 100

indices = torch.randperm(trainData:size(1)):long()
trainData = trainData:index(1,indices)
trainLabels = trainLabels:index(1,indices)

epochs = 40
print('Training Starting')
local optimParams = {learningRate = 0.01, learningRateDecay = 0.0001}
local _,loss
local losses = {}
for epoch=1,epochs do
    collectgarbage()
    print('Epoch '..epoch..'/'..epochs)
    for n=1,N-batchSize, batchSize do
        x = trainData:narrow(1,n,batchSize)
        y = trainLabels:narrow(1,n,batchSize)
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
    end
    --local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    --gnuplot.pngfigure('Training2.png')
    --gnuplot.plot(table.unpack(plots))
    --gnuplot.ylabel('Loss')
    --gnuplot.xlabel('Batch #')
    --gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData:size(1)):long()
    trainData = trainData:index(1,indices)
    trainLabels = trainLabels:index(1,indices)
end
print(losses)
torch.save('model_CNN.t7',model)
