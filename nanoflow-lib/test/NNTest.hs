module NNTest (test) where
import Test.Hspec
import Nanoflow.NN
import Data.Matrix

test :: Spec
test = do
    describe "NN Backpropagation" $ do
        it "checks that backpropagation works for a simple expression" $ do
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
            network <- inputLayer 1 >>= pushDALayer 1 ((-1), 1) htan  -- Generate Network
            let m = getElem 1 1(justValue . lweights . ltail $ network) -- Get m
            let b = getElem 1 1(justValue . lbiases  . ltail $ network) -- Get b
            let x = 3 -- Set x
            let c = 5 -- Set c
            let ff = feedforward [x] network  -- Feedforward
            let bp = Nanoflow.NN.backpropagate mse [c] ff -- Backpropagation
            let e = (tanh(m*x + b) - c)**2    -- Calculate correct error
            let n1 = tanh(m*x + b)            -- Calculate correct n1
            let n2 = m*x + b                  -- Calculate correct n2
            let dedn1 = 2*(n1 - c)            -- Calculate correct de/dn1
            let dedn2 = dedn1 * (1 / (cosh n2)**2) -- Calculate correct dn/dn2
            let dedm = dedn2 * x              -- Calculate correct de/dm
            let dedb = dedn2                  -- Calculate correct de/db
            -- Compare values with what we got in backpropagation
            let ededn1 = getElem 1 1 (justGrad . lnodes $ bp)
            let ededn2 = getElem 1 1 (justGrad . lnodes . ltail $ bp)
            let ededm  = getElem 1 1 (justGrad . lweights . ltail $ bp)
            let ededb  = getElem 1 1 (justGrad . lbiases . ltail $ bp)

            let threshold = 0.0001
            abs (ededn1 - dedn1) `shouldSatisfy` (< threshold)
            abs (ededn2 - dedn2) `shouldSatisfy` (< threshold)
            abs (ededm - dedm) `shouldSatisfy` (< threshold)
            abs (ededb - dedb) `shouldSatisfy` (< threshold)

            return ()
        return()
    describe "Complex Test" $ do
        it "checks that backpropagation works for a complex model" $ do
            network <- inputLayer 3 >>= pushDALayer 1 (1, 1) htan
            let ff = feedforward [1, 2, 3] network
            let bp = backpropagate mse [10] ff
            -- The model is:
            -- O = tanh (w1n1 + w2n2 + w3n3 + b), so the explicit error is
            -- e = (10 - tanh (w1n1 + w2n2 + w3n3 + b))**2
            --
            -- We calculate the derivatives, and obtain;
            -- de/dw1 = -2 * n1 * (10 - tanh (n1*w1 + n2 * w2 + n3*w3 + b)) / (cosh (n1*w1 + n2 * w2 + n3*w3 + b))**2
            let threshold = 0.0001
            let n1 = 1
            let n2 = 2
            let n3 = 3
            let w1 = 1
            let w2 = 1
            let w3 = 1
            let b = 1
            let correct = -2 * n1 * (10 - tanh (n1*w1 + n2 * w2 + n3*w3 + b)) / (cosh (n1*w1 + n2 * w2 + n3*w3 + b))**2
            let estimate = (getElem 1 1) . justGrad . lweights . ltail $ bp
            abs (correct - estimate) `shouldSatisfy` (< threshold)
            let correct = -2 * n2 * (10 - tanh (n1*w1 + n2 * w2 + n3*w3 + b)) / (cosh (n1*w1 + n2 * w2 + n3*w3 + b))**2
            let estimate = (getElem 1 2) . justGrad . lweights . ltail $ bp
            abs (correct - estimate) `shouldSatisfy` (< threshold)
            let correct = -2 * n3 * (10 - tanh (n1*w1 + n2 * w2 + n3*w3 + b)) / (cosh (n1*w1 + n2 * w2 + n3*w3 + b))**2
            let estimate = (getElem 1 3) . justGrad . lweights . ltail $ bp
            abs (correct - estimate) `shouldSatisfy` (< threshold)
            return ()
        return()
    describe "Complex Test" $ do
        it "checks that feedforward matches pytorch" $ do
            -- Results can be obtained from tools/pytorch_model_1.py
            -- Creates a network with 3 inputs, and one output. 
            -- tanh activation and a bias.
            let zeros = zeroM 1 1
            let nodes = zipParam zeros zeros
            let weights = zipParam (fromList 1 3 [0.1, 0.2, 0.3]) (fromList 1 3 [0, 0, 0])
            let biases = zipParam (fromList 1 1 [0.4]) (fromList 1 1 [0]) 
            inputLayer <- inputLayer 3
            let denseLayer = DenseLayer nodes weights biases inputLayer
            let network = ActivationLayer nodes htan denseLayer
            let ff = feedforward [1, 2, 3] network
            let estimate = (prediction ff) !! 0
            let correct = 0.9468
            let threshold = 0.01
            abs (correct - estimate) `shouldSatisfy` (< threshold)
            return()
    return()
    describe "Complex Test" $ do
        it "checks that backpropagation matches pytorch" $ do
            -- Results can be obtained from tools/pytorch_model_2.py
            -- Creates a network with 3 inputs, and one output. 
            -- tanh activation and a bias.
            let zeros = zeroM 1 1
            let nodes = zipParam zeros zeros
            let weights = zipParam (fromList 1 3 [0.1, 0.2, 0.3]) (fromList 1 3 [0, 0, 0])
            let biases = zipParam (fromList 1 1 [0.4]) (fromList 1 1 [0]) 
            inputLayer <- inputLayer 3
            let denseLayer = DenseLayer nodes weights biases inputLayer
            let network = ActivationLayer nodes htan denseLayer
            
            let training_set = [([1, 2, 3], [10]), ([4, 5, 6], [20])]
            let ag = accumulate_grad mse training_set network
            let output = updateWeights 1 ag
            let w = justValue . lweights . ltail $ output
            let w1 = getElem 1 1 w
            let w2 = getElem 1 2 w
            let w3 = getElem 1 3 w
            let b = (getElem 1 1) . justValue . lbiases . ltail $ output

            let w1' = 2.4284
            let w2' = 4.5167
            let w3' = 6.6051
            let b' = 2.3884

            let threshold = 0.01
            abs (w1 - w1') `shouldSatisfy` (< threshold)
            abs (w2 - w2') `shouldSatisfy` (< threshold)
            abs (w3 - w3') `shouldSatisfy` (< threshold)
            abs (b - b') `shouldSatisfy` (< threshold)

            return()
    return()
    describe "Complex Test" $ do
        it "checks that feedforward matches pytorch for a more complex model" $ do
            -- Results can be obtained from tools/pytorch_model_3.py
            -- Creates a network with 3 inputs, and one output. 
            -- tanh activation and a bias.
            let zeros = zeroM 3 1
            let nodes = zipParam zeros zeros
            let weights = zipParam (fromLists [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]) (fromList 3 3 [0, 0, 0, 0, 0, 0, 0, 0, 0])
            let biases = zipParam (fromList 3 1 [0.4, 0.5, 0.6]) (fromList 3 1 [0, 0, 0]) 
            inputLayer <- inputLayer 3
            let denseLayer = DenseLayer nodes weights biases inputLayer
            let network = ActivationLayer nodes htan denseLayer
            
            let input = [1, 2, 3]
            let ff = feedforward input network
            let pred = prediction $ ff
            let correct = [0.9468060, 0.9987782, 0.9999727]
            let threshold = 0.01
            fmse correct  pred `shouldSatisfy` (< threshold)
            return()
    return()
    describe "Complex Test" $ do
        it "checks that backpropagation matches pytorch for a more complex model" $ do
            -- Results can be obtained from tools/pytorch_model_4.py
            -- Creates a network with 3 inputs, and one output. 
            -- tanh activation and a bias.
            let zeros = zeroM 2 1
            let nodes = zipParam zeros zeros
            let weights = zipParam (fromLists [[0.1, 0.2], [0.3, 0.4]]) (fromList 2 2 [0, 0, 0, 0])
            let biases = zipParam (fromList 2 1 [0.4, 0.5]) (fromList 2 1 [0, 0]) 
            inputLayer <- inputLayer 2
            let denseLayer = DenseLayer nodes weights biases inputLayer
            let network = ActivationLayer nodes htan denseLayer
            
            let training_set = [([1, 2], [10, 11])]
            let ag = accumulate_grad mse training_set network
            let output = updateWeights 1 ag

            let threshold = 0.1

            let w = toList . justValue . lweights . ltail $ output
            let w' = [4.6204, 9.2408, 1.8171, 3.4341]
            fmse w w' `shouldSatisfy` (< threshold)

            let b = toList . justValue . lbiases . ltail $ output
            let b' = [4.9204, 2.0171]
            fmse b b' `shouldSatisfy` (< threshold)
            return()
    return()
    describe "Complex Test" $ do
        it "checks that backpropagation matches pytorch for an even more complex model" $ do
            -- Results can be obtained from tools/pytorch_model_5.py
            input <- inputLayer 2
            let nodes = zipParam (zeroM 64 1) (zeroM 64 1)
            let weights = zipParam (fromLists (replicate 64 [0.1, 0.1])) (fromLists (replicate 64 [0.0, 0.0]))
            let biases = zipParam (fromList 64 1 (replicate 64 0.1)) (fromList 64 1 (replicate 64 0.0))
            let hiddenLayer = DenseLayer nodes weights biases input
            hiddenLayer' <- pushActivationLayer relu hiddenLayer

            let nodes = zipParam (zeroM 1 1) (zeroM 1 1)
            let weights = zipParam (fromList 1 64 (replicate 64 0.1)) (fromList 1 64 (replicate 64 0.0))
            let biases = zipParam (fromList 1 1 [0.1]) (fromList 1 1 [0])

            let network' = DenseLayer nodes weights biases hiddenLayer'
            network <- pushActivationLayer sigmoid network'

            let ff = feedforward [1.0, 2.0] network
            let pred = (!!0) . prediction $ ff

            let threshold = 0.01
            abs (pred - 0.9346) `shouldSatisfy` (< threshold)

            let ag = accumulate_grad mse [([1.0, 2.0], [0.3])] network
            let step1 = updateWeights 1.0 ag
            
            let ff = feedforward [1.0, 2.0] step1
            let pred = (!!0) . prediction $ ff

            let threshold = 0.01
            abs (pred - 0.8296) `shouldSatisfy` (< threshold)

            return()
    return()
    describe "Complex Test" $ do
        it "checks that backpropagation matches pytorch for an even more complex model" $ do
            -- Results can be obtained from tools/pytorch_model_6.py
            input <- inputLayer 2
            let nodes = zipParam (zeroM 64 1) (zeroM 64 1)
            let weights = zipParam (fromLists (replicate 64 [0.1, 0.1])) (fromLists (replicate 64 [0.0, 0.0]))
            let biases = zipParam (fromList 64 1 (replicate 64 0.1)) (fromList 64 1 (replicate 64 0.0))
            let hiddenLayer = DenseLayer nodes weights biases input
            hiddenLayer' <- pushActivationLayer relu hiddenLayer

            let nodes = zipParam (zeroM 1 1) (zeroM 1 1)
            let weights = zipParam (fromList 1 64 (replicate 64 0.1)) (fromList 1 64 (replicate 64 0.0))
            let biases = zipParam (fromList 1 1 [0.1]) (fromList 1 1 [0])

            let network' = DenseLayer nodes weights biases hiddenLayer'
            network <- pushActivationLayer sigmoid network'

            let ff = feedforward [1.0, 2.0] network
            let pred = (!!0) . prediction $ ff

            let threshold = 0.01
            abs (pred - 0.9346) `shouldSatisfy` (< threshold)

            let ag = accumulate_grad mse [([1.0, 2.0], [0.3])] network
            let step1 = updateWeights 1.0 ag
            
            let ag = accumulate_grad mse [([1.0, 3.0], [0.4])] step1
            let step2 = updateWeights 1.0 ag
            
            let ff = feedforward [1.0, 2.0] step2
            let pred = (!!0) . prediction $ ff

            let threshold = 0.01
            abs (pred - 0.4316) `shouldSatisfy` (< threshold)

            return()
    return()
    describe "Complex Test" $ do
        it "checks that backpropagation matches pytorch for an even more complex model" $ do
            -- Results can be obtained from tools/pytorch_model_7.py
            input <- inputLayer 2
            let nodes = zipParam (zeroM 64 1) (zeroM 64 1)
            let weights = zipParam (fromLists (replicate 64 [0.1, 0.1])) (fromLists (replicate 64 [0.0, 0.0]))
            let biases = zipParam (fromList 64 1 (replicate 64 0.1)) (fromList 64 1 (replicate 64 0.0))
            let hiddenLayer = DenseLayer nodes weights biases input
            hiddenLayer' <- pushActivationLayer relu hiddenLayer

            let nodes = zipParam (zeroM 1 1) (zeroM 1 1)
            let weights = zipParam (fromList 1 64 (replicate 64 0.1)) (fromList 1 64 (replicate 64 0.0))
            let biases = zipParam (fromList 1 1 [0.1]) (fromList 1 1 [0])

            let network' = DenseLayer nodes weights biases hiddenLayer'
            network <- pushActivationLayer sigmoid network'

            let ff = feedforward [1.0, 2.0] network
            let pred = (!!0) . prediction $ ff

            let threshold = 0.01
            abs (pred - 0.9346) `shouldSatisfy` (< threshold)

            let ag = accumulate_grad mse [([1.0, 2.0], [0.5]), ([3.0, 4.0], [0.8])] network
            let output = updateWeights 0.5 ag
            
            let ff = feedforward [1.0, 2.0] output
            let pred = (!!0) . prediction $ ff

            let threshold = 0.01
            abs (pred - 0.9039) `shouldSatisfy` (< threshold)

            return()
    return()
