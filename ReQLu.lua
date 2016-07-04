local ReQLu, parent = torch.class('ReQLu', 'nn.Module')
-- transfer function f(x) = x^2 + x if x > 0 else 0

function ReQLu:__init()
  parent.__init(self)
  -- the two states module needs to maintain are outputs in forward and backward pass
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
end

-- define input to output mapping (forward pass)
function ReQLu:updateOutput(input)
  -- make sure the input is two dimensional ( batch_size x input_dimension)
  assert(input:nDimension() == 2)
  
  -- calculate output without mask
  self.output:resizeAs(input):copy(input)
  self.output:cmul(input):add(input)
  
  -- apply mask
  local mask = input:ge(0):typeAs(input)
  self.output:cmul(mask)
  return self.output
end

-- define gradOutput to gradInput mapping (backward pass)
function ReQLu:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  
  -- calculate dz/dx (without masking)
  self.gradInput:copy(2*input):add(torch.ones(input:size()))
  
  -- apply mask
  local mask = input:ge(0):typeAs(input)  -- convert from ByteTensor to Tensor
  self.gradInput:cmul(mask)
  
  -- calculate gradInput by multiplying it with gradOutput
  self.gradInput:cmul(gradOutput)
  return self.gradInput
end