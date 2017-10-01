require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

--totalNumLines
data_fer_file = io.open('./test/FER18.txt')
numLines = 0
for line in data_fer_file:lines() do
    numLines = numLines+1
end
print(numLines)
data_fer = torch.Tensor(numLines,6000)
data_fer_labels = torch.ones(numLines)
data_fer_file = io.open('./test/FER18.txt')
LineNum = 0;
for line in data_fer_file:lines() do
    LineNum=LineNum+1
    A = line:split(" ")
    if (#A) == 6000 then
        data_fer[LineNum]=torch.Tensor(A)
    end
end

data_infer_file = io.open('./test/INFR.txt')
numLines = 0
for line in data_infer_file:lines() do
numLines = numLines+1
end
print(numLines)
data_infer = torch.Tensor(numLines,6000)
data_infer_labels = torch.ones(numLines)*2
data_infer_file = io.open('./test/INFR.txt')
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

testData = torch.cat({data_fer,data_infer},1)
testlabels = torch.cat({data_fer_labels,data_infer_labels},1)
print(#testData)
print(#testlabels)

mean = testData:mean()
std = testData:std()
testData:add(-mean)
testData:div(std)
print(testData:mean())
print(testData:std())
testData = testData:reshape(testData:size(1),testData:size(2),1)

model = torch.load('model_CNN.t7')

N = testData:size(1)
teSize = N

print('Testing accuracy')
correct = 0
class_perform = {0,0}
class_size = {0,0}
classes = {'FER', 'INF'}
for i=1,N do
    local groundtruth = testlabels[i]
    local example = torch.Tensor(6000,1);
    example = testData[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end