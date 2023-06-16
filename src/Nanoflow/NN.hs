{-# LANGUAGE DeriveGeneric, DeriveAnyClass #-}

module Nanoflow.NN where
import Prelude hiding ((<*>))
import Control.Monad (replicateM)
import GHC.Generics (Generic)
import Control.DeepSeq
import System.Random (randomRIO)
import Data.Matrix

data Error = Error {efunc :: [Double] -> [Double] -> Double, egrad :: Matrix Double -> Matrix Double -> Matrix Double}
data Activator = Activator {afunc :: Double -> Double, agrad :: Double -> Double, aname :: String} deriving (Generic, NFData)
data Parameter = Parameter {pvalue :: Double, pgrad :: Double} deriving (Show, Generic, NFData)
data Network = InputLayer  {lnodes :: Matrix Parameter} |
               HiddenLayer {lnodes :: Matrix Parameter, lweights :: Matrix Parameter, lbiases :: Matrix Parameter, ltail :: Network} |
               ActivationLayer {lnodes :: Matrix Parameter, lactivator :: Activator, ltail :: Network} deriving (Generic, NFData)

instance Show Activator where
    show (Activator _ _ name) = show name



-- Generate n random values between (low, max)
runif :: Int -> (Double, Double) -> IO [Double]
runif n (l, h) = do
  randomValues <- replicateM n (randomRIO (l, h) :: IO Double)
  return randomValues

-- Generates a random matrix of size (i, j) with values between (0, 1)
randM :: Int -> Int -> IO (Matrix Double)
randM rows cols = do
    values <- runif (rows*cols) (0, 1.0)
    let output = fromList rows cols values
    return output

-- Generates a matrix of size (i,j) with values equal to 0
zeroM :: Int -> Int -> Matrix Double
zeroM rows cols = fromList rows cols (replicate (rows*cols) 0)

-- Size of a matrix
size :: Matrix a -> (Int, Int)
size m = (nrows m, ncols m)


(<*>) :: Floating a => Matrix a -> Matrix a -> Matrix a
(<*>) mat1 mat2 
    | size mat1 == size mat2 = output
    | otherwise = error "Dimension mismatch" where
    output = matrix (nrows mat1) (ncols mat2) (\(i,j) -> ((mat1 ! (i,j)) * (mat2 ! (i, j))))

-- (Double, Double) <-> Parameter conversion
parameter :: (Double, Double) -> Parameter
parameter (val, grad) = Parameter val grad

justValue :: Matrix Parameter -> Matrix Double
justValue = fmap pvalue

justGrad  :: Matrix Parameter -> Matrix Double
justGrad = fmap pgrad

zipM :: Matrix a -> Matrix b -> Matrix (a, b)
zipM m n
    | size m == size n = mzipn
    | otherwise = error "Dimension mismatch" where
    (rows, cols) = size m
    fm = toList m
    fn = toList n
    fmzipfn = zip fm fn
    mzipn = fromList rows cols fmzipfn


zipParam :: Matrix Double -> Matrix Double -> Matrix Parameter
zipParam values grads = fmap parameter (zipM values grads)

inputLayer :: Int -> IO Network
inputLayer size = do
    -- Generate zero nodes
    let zeros = zeroM size 1
    let nodes = zipParam zeros zeros
    return $ (InputLayer nodes)

pushWeightLayer :: Int -> Network -> IO Network
pushWeightLayer size net = do
    -- Generate zero nodes
    let zeros = zeroM size 1
    let nodes = zipParam zeros zeros
    
    -- Generate Random weights
    let rows = size :: Int
    let cols = nrows (lnodes net) :: Int
    weight_values <- randM rows cols
    let weight_grads = zeroM rows cols
    let weights = zipParam weight_values weight_grads

    -- Generate Random Biases
    bias_values <- randM size 1
    let bias_grads = zeroM size 1
    let bias = zipParam bias_values bias_grads
    
    return $ (HiddenLayer nodes weights bias net)

pushActivationLayer :: Activator -> Network -> IO Network
pushActivationLayer activator net = do
    -- Generate zero nodes
    let rows = nrows (lnodes net) :: Int
    let zeros = zeroM rows 1
    let nodes = zipParam zeros zeros

    return $ (ActivationLayer nodes activator net)

pushLayer :: Int -> Activator -> Network -> IO Network
pushLayer size actv net = pushWeightLayer size net >>= pushActivationLayer actv

pzerograd :: Parameter -> Parameter
pzerograd (Parameter val _) = Parameter val 0

zerograd :: Network -> Network
zerograd (InputLayer nodes) = InputLayer (fmap pzerograd nodes)
zerograd (HiddenLayer nodes weights biases tail) = HiddenLayer (fmap pzerograd nodes) (fmap pzerograd weights) (fmap pzerograd biases) (zerograd tail)
zerograd (ActivationLayer nodes activator tail) = ActivationLayer (fmap pzerograd nodes) activator (zerograd tail)

inputSize :: Network -> Int 
inputSize (InputLayer nodes) = nrows nodes
inputSize (HiddenLayer _ _ _ tail) = inputSize tail
inputSize (ActivationLayer _ _ tail) = inputSize tail


feedforward input (InputLayer nodes) 
    | nrows nodes == rows = output
    | otherwise = error "Dimension mismatch." where
        rows = length input :: Int;
        values = fromList rows 1 input
        grads = justGrad nodes
        newnodes = zipParam values grads
        output = InputLayer newnodes

feedforward input (HiddenLayer nodes weights biases tail) = output where
    newtail = feedforward input tail
    x = justValue . lnodes $ newtail
    m = justValue weights 
    b = justValue biases
    values = m * x + b
    grads = justGrad nodes
    newnodes = zipParam values grads
    output = HiddenLayer newnodes weights biases newtail

feedforward input (ActivationLayer nodes activator tail) = output where
    newtail = feedforward input tail
    x = justValue . lnodes $ newtail
    values = fmap (afunc activator) x
    grads = justGrad nodes
    newnodes = zipParam values grads
    output = ActivationLayer newnodes activator newtail
    
addGrad :: Matrix Parameter -> Matrix Double -> Matrix Parameter
addGrad original diff = output where
    values = justValue original
    grads = justGrad original
    output = zipParam values (grads + diff)

addValue :: Matrix Parameter -> Matrix Double -> Matrix Parameter
addValue original diff = output where
    values = justValue original
    grads = justGrad original
    output = zipParam (values + diff) grads

-- A Layer contains a set of nodes y, a set of Weights W and a set of biases b such that
-- y = W x + b, 
-- where x is the set of nodes of the previous layer.
--
-- Under this notation, we calculate the gradient as follows:
-- grad(W) = grad(y) * t(x)
-- grad(b) = grad(y)
-- grad(x) = t(W) * grad(y)
--
-- For more information: https://mlvu.github.io/

xbackpropagate :: Matrix Double -> Network -> Network
xbackpropagate grads (InputLayer nodes) = output where
    newnodes = addGrad nodes grads
    output = InputLayer newnodes

xbackpropagate grads (HiddenLayer nodes weights biases tail) = output where
    newnodes = addGrad nodes grads
    newbiases = addGrad biases grads
    
    x = justValue . lnodes $ tail
    newweights = addGrad weights (grads * (transpose x))

    delta_x = (transpose (justValue weights)) * grads
    newtail =  xbackpropagate delta_x tail

    output = HiddenLayer newnodes newweights newbiases newtail

xbackpropagate grads (ActivationLayer nodes activator tail) = output where
    newnodes = addGrad nodes grads
    x = justValue . lnodes $ tail
    dx = fmap (agrad activator) x
    delta_x = grads <*> dx 
    newtail = xbackpropagate delta_x tail
    output = ActivationLayer newnodes activator newtail

backpropagate :: Error -> [Double] -> Network -> Network
backpropagate err lexpected_value net = xbackpropagate grads net where
    predicted_value = justValue . lnodes $ net
    expected_value = fromList (nrows predicted_value) (ncols predicted_value) lexpected_value
    grads = (egrad err) predicted_value expected_value


accumulate_grad :: Error -> [([Double], [Double])] -> Network -> Network
accumulate_grad err [] net = net
accumulate_grad err (t:ts) net = accumulate_grad err ts bpnet where
    (input, correct) = t
    ffnet = feedforward input net
    bpnet = backpropagate err correct ffnet
   
deviation :: Error -> [Double] -> Network -> Double
deviation err lexpected_value net = deviation where
    predicted_value = justValue . lnodes $ net
    lpredicted_value = toList predicted_value
    deviation = (efunc err) lpredicted_value lexpected_value

prediction :: Network -> [Double]
prediction = toList . justValue . lnodes

updateWeights :: Double -> Network -> Network
updateWeights rate (InputLayer x) = (InputLayer x)
updateWeights rate (ActivationLayer nodes actv tail) = ActivationLayer nodes actv (updateWeights rate tail)
updateWeights rate (HiddenLayer nodes weights biases tail) = output where
    delta_weight = fmap (*(-rate)) (justGrad weights)
    newweights = addValue weights delta_weight
    delta_bias = fmap (*(-rate)) (justGrad biases)
    newbiases = addValue biases delta_bias
    newtail = updateWeights rate tail
    output = HiddenLayer nodes newweights newbiases newtail


printNetwork :: Network -> String
printNetwork (InputLayer nodes) = 
    "Input Layer\n" ++
    "Nodes\n" ++
    show (justValue nodes) ++ "\n" ++
    show (justGrad nodes) ++ "\n\n"

printNetwork (HiddenLayer nodes weights bias tail) = 
    printNetwork tail ++
    "Hidden Network\n" ++ 
    "\nWeights\n" ++
    show (justValue weights) ++ "\n" ++
    show (justGrad weights) ++ "\n" ++
    "\nBias\n" ++
    show (justValue bias) ++ "\n" ++
    show (justGrad bias) ++ "\n" ++
    "Nodes\n" ++
    show (justValue nodes) ++ "\n" ++
    show (justGrad nodes) ++ "\n"

printNetwork (ActivationLayer nodes activator tail) =
    printNetwork tail ++
    "Activation function\n" ++
    "Function: " ++ (show . aname $ activator) ++ "\n" ++
    "Nodes\n" ++
    show (justValue nodes) ++ "\n" ++
    show (justGrad nodes) ++ "\n"


instance Show Network where
    show = printNetwork

-- Activator and Error functions
--
-- tanh
dtanh :: Floating a => a -> a
dtanh x = (1/(cosh x))**2

htan :: Activator 
htan = Activator Prelude.tanh dtanh "tanh"

-- Identity 
idd :: Activator
idd = Activator id (\x -> 1) "id"

-- Mean squared Error
fmse :: [Double] -> [Double] -> Double
fmse xs ys
  | length xs /= length ys = error "Input lists must have the same length"
  | otherwise = sum [(x - y) ^ 2 | (x, y) <- zip xs ys] / fromIntegral (length xs)

dmse :: Matrix Double -> Matrix Double -> Matrix Double
dmse value expected = fmap (2*) (value - expected)

mse :: Error
mse = Error fmse dmse

