-- The model is
-- x ---> mx + b ---> tanh(mx + b) ---> (tanh(mx + b) - c)**2
--
-- So the error function is explicitly defined as:
--  e = (tanh(mx +b) - c)**2
--
-- With these, we calculate:
-- Let 
--  e = (n1 - c)**2
--  n1 = tanh(mx + b) = tanh(n2)
--  n2 = mx + b
-- 
-- We derive:
--
-- de/dn1 = 2*(n1 - c)
-- de/dn2 = de/dn1 * dn1/dn2 = 2*(n1 - c) * (1 / (cosh(n2)**2))
-- de/dm  = de/dn2 * dn2/dm  = 2*(n1 - c) * (1 / (cosh(n2)**2)) * x
-- de/db  = de/dn2 * dn2/db  = 2*(n1 - c) * (1 / (cosh(n2)**2)) 
--
-- 
network <- inputLayer 1 >>= pushLayer 1 htan  -- Generate Network
m = getElem 1 1(justValue . weights . ntail $ network) -- Get m
b = getElem 1 1(justValue . biases  . ntail $ network) -- Get b
x = 3 -- Set x
c = 5 -- Set c
ff = feedforward [x] network  -- Feedforward
bp = backpropagate mse [c] ff -- Backpropagation
e = (tanh(m*x + b) - c)**2    -- Calculate correct error
n1 = tanh(m*x + b)            -- Calculate correct n1
n2 = m*x + b                  -- Calculate correct n2
dedn1 = 2*(n1 - c)            -- Calculate correct de/dn1
dedn2 = dedn1 * (1 / (cosh n2)**2) -- Calculate correct dn/dn2
dedm = dedn2 * x              -- Calculate correct de/dm
dedb = dedn2                  -- Calculate correct de/db
-- Compare values with what we got in backpropagation



network <- inputLayer 1 >>= pushLayer 1 htan
ff = feedforward [0.3] network
bp = backpropagate mse [0.5] ff
deviation mse [0.5] bp
uw = updateWeights 0.1 bp
--loop
network = zerograd uw
ff = feedforward [0.3] network
bp = backpropagate mse [0.5] ff
deviation mse [0.5] bp
uw = updateWeights 0.01 bp


