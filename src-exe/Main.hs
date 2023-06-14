module Main where
import Nanoflow
import Data.Matrix
import System.Random
import Text.Printf (printf)


generateTrainingSet :: Int -> (Double, Double) -> Network -> IO [(Matrix Double, Matrix Double)]
generateTrainingSet n (min, max) net 
    | n <= 0 = return []
    | otherwise = do
        let input_size = inputSize net
        linput <- runif input_size (min, max)
        let input = fromList input_size 1 linput
        let pred = prediction (feedforward linput net)
        tail <- generateTrainingSet (n - 1) (min, max) net
        let output = [(input, pred)] ++ tail
        return $ output

overall_error :: Error -> [(Matrix Double, Matrix Double)] -> Network -> Double
overall_error err [] net = 0
overall_error err (t:ts) net = this + rest where
    (input,  correct) = t
    (linput, lcorrect)  = (toList input, toList correct)
    ff = feedforward linput net
    this = deviation err lcorrect ff
    rest = overall_error err ts ff

accumulate_grad :: Error -> [(Matrix Double, Matrix Double)] -> Network -> Network
accumulate_grad err [] net = net
accumulate_grad err (t:ts) net = accumulate_grad err ts bpnet where
    (input, correct) = t
    (linput, lcorrect) = (toList input, toList correct)
    ffnet = feedforward linput net
    bpnet = backpropagate err lcorrect ffnet

loop :: Int -> Double -> Network -> [(Matrix Double, Matrix Double)] -> IO Network
loop n rate net training_set
    | n < 0 = return net
    | otherwise = do
        let l = fromIntegral (length training_set) :: Double
        let error = (1/l) * (overall_error mse training_set net)
        printf "Iteration: %03d Error: %.7f\n" n error


        let zgnet = zerograd net
        let gnet = accumulate_grad mse training_set zgnet
        let newnet = updateWeights rate gnet


        output <- loop (n - 1) rate newnet training_set 
        return $ output

main :: IO ()
main = do
  network <- inputLayer 1 >>= pushLayer 16 htan >>= pushLayer 16 htan >>= pushLayer 1 idd
  training_set <- generateTrainingSet 10 (0.0, 1.0) network

  model <- inputLayer 1 >>= pushLayer 16 htan >>= pushLayer 16 htan >>= pushLayer 1 idd
  loop 300 (0.0001) model training_set
  putStrLn "Over!"
