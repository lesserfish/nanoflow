network <- inputLayer 1 >>= pushLayer 1 htan
m = getElem 1 1(justValue . weights . ntail $ network) 
b = getElem 1 1(justValue . biases  . ntail $ network) 
x = 3
c = 5
px = [x]
pc = [c]
ff = feedforward px network 
bp = backpropagate mse pc ff
e = (tanh(m*x + b) - c)**2
n1 = tanh(m*x + b)
dedn1 = 2*(n1 - c)
n2 = m*x + b
dedn2 = dedn1 * (1 / (cosh n2)**2)
dedm = dedn2 * x
dedb = dedn2



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


