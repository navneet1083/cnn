require 'nn'

torch.manualSeed(1234)

local model = nn.Sequential()
local n = 2 -- two input feature
local k = 1  -- scalar output (only one output)
local s = {n, 10, k} -- {10 -- hidden layers, so it's a 3-layer neural network }

model:add(nn.Linear(s[1],s[2]))
-- Add non-Linearity
model:add(nn.Tanh())

model:add(nn.Linear(s[2], s[3]))
model:add(nn.Tanh())

-- Defining Loss function
local loss = nn.MSECriterion()

-- building dataset
local m = 128
local X = torch.DoubleTensor(m,n) -- cudaTensor with 'mxn' dimension
local Y = torch.DoubleTensor(m) -- CudaTensor


for i = 1,m do
  local x = torch.randn(2) -- normal distribution of 2 elements
  local y = x[1] * x[2] > 0 and -1 or 1 -- both are positive or negative
  X[i]:copy(x)  -- fine also for cuda
  Y[i] = y   -- fine also for Cuda
end

-- global variable; checking for interactive mode.
-- e.g th -i train.lua
-- net = model

-- GPU computation
require 'cunn'
model:cuda()
loss:cuda()
X = X:cuda()
Y = Y:cuda()



local theta, gradTheta = model:getParameters() -- will give 1d theta

-- All parameters which will sent to optimizer
local optimState = {learningRate = 0.15}



-- start training

require 'optim'

for epoch = 1, 1e3 do
    function feval(theat)
        gradTheta:zero()
	-- model:zeroGradParameters()
	local h_x = model:forward(X)
	local J = loss:forward(h_x, Y)
	print(J) -- just for debugging
	local dJ_dh_x = loss:backward(h_x, Y)
	model:backward(X, dJ_dh_x) -- This computes and updates gradTheta
	return J, gradTheta
    end
    optim.sgd(feval, theta, optimState)
end


print('Prev: 0.056')
net = model



	





