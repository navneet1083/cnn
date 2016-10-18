require 'nn';
require 'pretty-nn';


local net = nn.Sequential()

-- Add First Layer
-- 3 - input channels
-- 6 - output channels (as we are creating channels from 3-to-6)
-- 5,5 - kernel size of 5-by-5
-- 1,1 - stride of 1-by-1
-- 2,2 - padding of 2-by-2
net:add(nn.SpatialConvolution(3, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())

-- Add Second Layer
-- Now, the input channel will be 6 and again output channel will be 6
--    rest will be same (5,5 kernel; 1,1 stride; 2,2 padding)
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())

-- Add Third Layer
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())

-- Add View
net:add(nn.View(-1))

-- Final Output, before this get the size of the forward
-- e.g x = torch.Tensor(3, 256, 256)
--     net:forward(x)
--     #net
net:add(nn.Linear(393216, 1000))

return net