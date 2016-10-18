-- My First CNN + Pool
require 'nn'




local net = nn.Sequential()

-- Add First Layer
-- 3 - input channels
-- 6 - output channels (as we are creating channels from 3-to-6)
-- 5,5 - kernel size of 5-by-5
-- 2,2 - stride of 2-by-2
-- 2,2 - padding of 2-by-2
net:add(nn.SpatialConvolution(3, 6, 5, 5, 2, 2, 2, 2))
net:add(nn.ReLU())

-- Add Second Layer
-- Now, the input channel will be 6 and again output channel will be 6
--    rest will be same (5,5 kernel; 1,1 stride; 2,2 padding)
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())

-- Add Third Layer
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())

-- Adding pooling
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-- Add View
net:add(nn.View(-1))

net:add(nn.Linear(6144, 1000))



return net