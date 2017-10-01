--[[
Model file for CNN for classifying window around TSS into fertile or infertile subjects.
--]]

require 'nn';

model = nn.Sequential()
model:add(nn.TemporalConvolution(1,32,11,4)) --1
model:add(nn.ReLU()) --2
model:add(nn.TemporalMaxPooling(2,2)) --3

model:add(nn.TemporalConvolution(32,96,5,2)) --4
model:add(nn.ReLU()) --5
model:add(nn.TemporalMaxPooling(2,2)) --6

model:add(nn.TemporalConvolution(96,256,5,2)) --7
model:add(nn.ReLU()) --8
model:add(nn.TemporalMaxPooling(2,2)) --9

model:add(nn.TemporalConvolution(256,256,5,2)) --10
model:add(nn.ReLU()) --11
model:add(nn.TemporalMaxPooling(2,2)) --12

model:add(nn.View(-1):setNumInputDims(2)) --13
model:add(nn.Linear(2560,500)) --14
model:add(nn.ReLU()) --15
model:add(nn.Dropout(0.5)) --16
model:add(nn.Linear(500,50)) --17
model:add(nn.ReLU()) --18
model:add(nn.Dropout(0.5)) --19
model:add(nn.Linear(50,2)) --20
model:add(nn.LogSoftMax())-- 21

return model
