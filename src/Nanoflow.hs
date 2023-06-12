module Nanoflow where
import Control.Monad (replicateM)
import Data.Matrix
import System.Random

data Activator = Activator {function :: Float -> Float, derivative :: Float -> Float, name :: String}
data Parameter = Parameter {value :: Float, grad :: Float} deriving Show
data Network = InputLayer  {nodes :: Matrix Parameter} |
               HiddenLayer {nodes :: Matrix Parameter, weights :: Matrix Parameter, biases :: Matrix Parameter, ntail :: Network} |
               ActivationLayer {nodes :: Matrix Parameter, activator :: Activator, ntail :: Network} deriving Show

instance Show Activator where
    show (Activator _ _ name) = show name

runif :: Int -> (Float, Float) -> IO [Float]
runif n (l, h) = do
  randomValues <- replicateM n (randomRIO (l, h) :: IO Float)
  return randomValues

parameter :: (Float, Float) -> Parameter
parameter (val, grad) = Parameter val grad

sech :: Floating a => a -> a
sech x = 1 / cosh x

dtanh :: Floating a => a -> a
dtanh x = (1/(cosh x))**2

htan :: Activator 
htan = Activator Prelude.tanh dtanh "tanh"

hadamardProduct :: Floating a => Matrix a -> Matrix a -> Matrix a
hadamardProduct mat1 mat2 
    | (nrows mat1, ncols mat1) == (nrows mat2, ncols mat2) = _result
    | otherwise = error "Dimension mismatch" where
    _result = matrix (nrows mat1) (ncols mat2) (\(i,j) -> ((mat1 ! (i,j)) * (mat2 ! (i, j))))

justValue :: Matrix Parameter -> Matrix Float
justValue = fmap value

justGrad  :: Matrix Parameter -> Matrix Float
justGrad = fmap grad

combineParam :: Matrix Float -> Matrix Float -> Matrix Parameter
combineParam vals grads
    | (nrows vals, ncols vals) == (nrows grads, ncols grads) = _result
    | otherwise = error "Dimension mismatch" where
        nrow = nrows grads :: Int
        ncol = ncols grads :: Int
        _lvals = toList vals :: [Float]
        _lgrads = toList grads :: [Float]
        _zip = zip _lvals _lgrads :: [(Float, Float)]
        _params = map parameter _zip :: [Parameter]
        _result = fromList nrow ncol _params

inputLayer :: Int -> IO Network
inputLayer _size = do
    -- Generate Random Nodes
    _node_values <- runif _size ((-1), 1) :: IO [Float]
    let _zeros = replicate _size 0 :: [Float]
    let _lzip = zip _node_values _zeros :: [(Float, Float)]
    let _pnodes = map parameter _lzip :: [Parameter]
    let _nodes = fromList _size 1 _pnodes :: Matrix Parameter
    let _output = InputLayer _nodes
    return $ _output

pushWeightLayer :: Int -> Network -> IO Network
pushWeightLayer _size _net = do
    -- Generate Random nodes
    _node_values <- runif _size ((-1), 1) :: IO [Float]
    let _zeros = replicate _size 0 :: [Float]
    let _lzip = zip _node_values _zeros :: [(Float, Float)]
    let _pnodes = map parameter _lzip :: [Parameter]
    let _nodes = fromList _size 1 _pnodes :: Matrix Parameter
    
    -- Generate Random weights
    let _nrow = _size :: Int
    let _ncol = nrows (nodes _net) :: Int
    _weight_values <- runif (_nrow*_ncol) ((-1), 1) :: IO [Float]
    let _wzeros = replicate (_nrow*_ncol) 0 :: [Float]
    let _wzip = zip _weight_values _wzeros :: [(Float, Float)]
    let _pweights = map parameter _wzip :: [Parameter]
    let _weights = fromList _nrow _ncol _pweights :: Matrix Parameter

    -- Generate Random Biases
    _bias_values <- runif _size ((-1), 1) :: IO [Float]
    let _bzeros = replicate _size 0 :: [Float]
    let _bzip = zip _bias_values _bzeros :: [(Float, Float)]
    let _pbias = map parameter _bzip :: [Parameter]
    let _bias = fromList _size 1 _pbias :: Matrix Parameter
    

    let _output = HiddenLayer _nodes _weights _bias _net
    return $ _output

pushActivationLayer :: Activator -> Network -> IO Network
pushActivationLayer _actv _net = do
    let _nrow = nrows (nodes _net) :: Int
    _node_values <- runif _nrow ((-1), 1) :: IO [Float]
    let _zeros = replicate _nrow 0 :: [Float]
    let _lzip = zip _node_values _zeros :: [(Float, Float)]
    let _pnodes = map parameter _lzip :: [Parameter]
    let _nodes = fromList _nrow 1 _pnodes :: Matrix Parameter

    let _output = ActivationLayer _nodes _actv _net
    return _output

pushLayer :: Int -> Activator -> Network -> IO Network
pushLayer _size _actv _net = pushWeightLayer _size _net >>= pushActivationLayer _actv

pzerograd :: Parameter -> Parameter
pzerograd (Parameter val _) = Parameter val 0

zerograd :: Network -> Network
zerograd (InputLayer _nodes) = InputLayer (fmap pzerograd _nodes)
zerograd (HiddenLayer _nodes _weights _biases _tail) = HiddenLayer (fmap pzerograd _nodes) (fmap pzerograd _weights) (fmap pzerograd _biases) (zerograd _tail)

feedforward :: [Float] -> Network -> Network
feedforward _input (InputLayer _nodes) 
    | nrows _nodes == nrow = InputLayer _nodes
    | otherwise = error "Dimension mismatch." where
        nrow = length _input :: Int;
        _zeros = zero nrow 1 :: Matrix Float
        _minput = fromList nrow 1 _input :: Matrix Float
        _nodes = combineParam _minput _zeros

feedforward _input (HiddenLayer _nodes _weights _biases _tail) = _result where
    _newtail = feedforward _input _tail :: Network
    _x = justValue . nodes $ _newtail :: Matrix Float
    _M = justValue _weights :: Matrix Float
    _b = justValue _biases :: Matrix Float
    _y = _M * _x + _b :: Matrix Float
    _grads = justGrad _nodes :: Matrix Float
    _newnodes = combineParam _y _grads :: Matrix Parameter
    _result = HiddenLayer _newnodes _weights _biases _newtail

feedforward _input (ActivationLayer _nodes _activator _tail) = _result where
    _newtail = feedforward _input _tail :: Network
    _x = justValue . nodes $ _newtail :: Matrix Float
    _y = fmap (function _activator) _x :: Matrix Float
    _grads = justGrad _nodes :: Matrix Float
    _newnodes = combineParam _y _grads :: Matrix Parameter
    _result = ActivationLayer _newnodes _activator _newtail
    


-- A Layer contains a set of nodes y, a set of Weights W and a set of biases b such that
-- y = W x + b, 
-- where x is the set of nodes of the previous layer.
--
-- Under this notation, we calculate the gradient as follows:
-- grad(W) = grad(y) * t(x)
-- grad(b) = grad(y)
-- grad(x) = t(W) * grad(y)
--
-- xbackpropagate receives as input grad(y) and the network. 
-- It then calculates grad(w), grad(b).
-- Then, it recurisvely calculates grad(x) backpropagating through the network.
--
-- For more information: https://mlvu.github.io/

xbackpropagate :: Matrix Float -> Network -> Network
xbackpropagate _grads (InputLayer _nodes) = _result where
    _values = justValue _nodes
    _original_grads = justGrad _nodes
    _newnodes = combineParam _values  (_original_grads + _grads)
    _result = InputLayer _newnodes

xbackpropagate _grads (HiddenLayer _nodes _weights _biases _tail) = _result where
    _biases_values = justValue _biases :: Matrix Float
    _biases_weights = justGrad _biases :: Matrix Float
    _newbiases = combineParam _biases_values (_biases_weights + _grads) :: Matrix Parameter

    _weight_values = justValue _weights :: Matrix Float
    _weight_grads_delta = _grads * (transpose _x) :: Matrix Float
    _x = justValue . nodes $ _tail :: Matrix Float
    _weight_grads = justGrad _weights :: Matrix Float
    _newweights = combineParam _weight_values (_weight_grads + _weight_grads_delta)

    _x_grads = (transpose _weight_values) * _grads
    _newtail = xbackpropagate _x_grads _tail
    _result = HiddenLayer _nodes _newweights _newbiases _newtail

xbackpropagate _grads (ActivationLayer _nodes _activator _tail) = _result where
    _values = justValue _nodes
    _weights = justGrad _nodes
    _newnodes = combineParam _values (_weights + _grads)

    _x = justValue . nodes $ _tail :: Matrix Float
    _Dx = fmap (derivative _activator) _x :: Matrix Float
    _x_grads = hadamardProduct _grads _Dx :: Matrix Float
    _newtail = xbackpropagate _x_grads _tail :: Network
    _result = ActivationLayer _nodes _activator _newtail :: Network

backpropagate :: Network -> Network
backpropagate _net = xbackpropagate _grads _net where
    nrow = nrows . nodes $ _net
    _value = replicate nrow 1
    _grads = fromList nrow 1 _value

updateWeights :: Float -> Network -> Network
updateWeights = undefined

-- Loop:
-- zerograd()
-- For t in Training Data:
--      network = feedforward network t
--      network = backpropagate network
-- network = updateWeights network rate
--
--
-- TODO: Rename every variable used in every function. The code, as it stands now, is unreadable.
--
