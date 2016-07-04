local ReQLuScaled, parent = torch.class('ReQLuScaled', 'nn.Module')
-- transfer function f(x) = a*x^2 + b*x if x > 0 else 0

function ReQLuScaled:__init()
  parent.__init(self)
  -- the two states module needs to maintain are outputs in forward and backward pass
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()

  -- declare weights  
  self.a = torch.Tensor(1)
  self.b = torch.Tensor(1)
  
  -- declare grad of weights
  self.grad_a = torch.Tensor(1)
  self.grad_b = torch.Tensor(1)
end

-- define input to output mapping (forward pass)
function ReQLuScaled:updateOutput(input)
  -- make sure the input is two dimensional ( batch_size x input_dimension)
  assert(input:nDimension() == 2)
  
  -- calculate output without mask
  self.output:resizeAs(input):copy(input)
  self.output:cmul(self.a[1] * input)
  self.output:add(self.b[1] * input)
  
  -- apply mask
  local mask = input:gt(0):typeAs(input)
  self.output:cmul(mask)
  return self.output
end

-- define gradOutput to gradInput mapping (backward pass)
function ReQLuScaled:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  
  -- calculate dz/dx (without masking)
  self.gradInput:copy(2*self.a[1] * input):add(self.b[1] * torch.ones(input:size()))
  
  -- apply mask
  local mask = input:gt(0):typeAs(input)  -- convert from ByteTensor to Tensor
  self.gradInput:cmul(mask)
  
  -- calculate gradInput by multiplying it with gradOutput
  self.gradInput:cmul(gradOutput)
  return self.gradInput
end


function ReQLuScaled:accGradParameters(input, gradOutput)
  -- calculate gradient wrt output
  local grad_a = torch.cmul(input, input)
  local grad_b = input
  
  -- apply mask
  local mask = input:gt(0):typeAs(input)  -- convert from ByteTensor to Tensor
  grad_a:cmul(mask)
  grad_b:cmul(mask)
  
  -- multiply by gradOutput
  grad_a:cmul(gradOutput)
  grad_b:cmul(gradOutput)
  
  -- update gradients
  self.grad_a = torch.sum(grad_a)
  self.grad_b = torch.sum(grad_b)
end

-- override the parameters function 
function ReQLuScaled:parameters()
  local weights = {self.a, self.b}
  local gradWeights =  {self.grad_a, self.grad_b}
  return weights, gradWeights
end