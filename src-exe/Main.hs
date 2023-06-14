module Main where
import Nanoflow
import Data.Matrix
import System.Random
import Text.Printf (printf)


add_noise :: Double -> [Double] -> IO [Double]
add_noise epsilon values = do
    noise <- runif (length values) ((-epsilon), epsilon)
    let output = zipWith (+) values noise
    return output

make_moon :: Int -> Double -> IO [(Matrix Double, Matrix Double)]
make_moon n epsilon = do
    let step_size = pi/(fromIntegral n) :: Double
    let steps = [0, 0 + step_size .. pi]
    let outer' = [[cos alpha, sin alpha] | alpha <- steps] :: [[Double]]
    let inner' = [[1 - cos alpha, 0.5 - sin alpha] | alpha <- steps] :: [[Double]]
    outer <- mapM (add_noise 0.1) outer'
    inner <- mapM (add_noise 0.1) inner'
    let outer_position = fmap (fromList 2 1) outer
    let outer_class = replicate (length outer_position) (fromList 1 1 [0])
    let inner_position = fmap (fromList 2 1) inner
    let inner_class = replicate (length inner_position) (fromList 1 1 [1])
    let outer_object = zip outer_position outer_class :: [(Matrix Double, Matrix Double)]
    let inner_object = zip inner_position inner_class :: [(Matrix Double, Matrix Double)]
    let output = outer_object ++ inner_object
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

getAccuracy :: Network -> [(Matrix Double, Matrix Double)] -> IO Double
getAccuracy network [] = return 0
getAccuracy network (y:ys) = do
    let (input, output) = y :: (Matrix Double, Matrix Double)
    let pred' = getElem 1 1 (prediction (feedforward (toList input) network)) :: Double
    let pred = if pred' > 0.5 then 1 else 0
    let hit = if pred == (getElem 1 1 output) then 1 else 0
    rest <- getAccuracy network ys
    return $ hit + rest

main :: IO ()
main = do
  training_set <- make_moon 100 0.01
  model <- inputLayer 2 >>= pushLayer 6 htan >>= pushLayer 6 htan >>= pushLayer 1 htan
  result <- loop 500 (0.001) model training_set
  acc <- (getAccuracy result training_set)
  let proportion = acc / (fromIntegral . length $ training_set) :: Double
  putStrLn $ "Accuracy: " ++ show proportion
  return ()
