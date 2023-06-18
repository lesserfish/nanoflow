module Demos.NNMoonsPlot where
import Nanoflow.NN
import Data.Matrix
import System.Random
import Control.DeepSeq
import Text.Printf (printf)
import Control.Monad
import Graphics.Matplotlib
import System.IO (hFlush, stdout)

add_noise :: Double -> [Double] -> IO [Double]
add_noise epsilon values = do
    noise <- runif (length values) ((-epsilon), epsilon)
    let output = zipWith (+) values noise
    return output

make_moon :: Int -> Double -> IO [([Double], [Double])]
make_moon n epsilon = do
    let step_size = pi/(fromIntegral n) :: Double
    let steps = [0, 0 + step_size .. pi]
    let outer' = [[cos alpha, sin alpha] | alpha <- steps] :: [[Double]]
    let inner' = [[1 - cos alpha, 0.5 - sin alpha] | alpha <- steps] :: [[Double]]
    outer <- mapM (add_noise epsilon) outer'
    inner <- mapM (add_noise epsilon) inner'
    let outer_position =  outer
    let outer_class = replicate (length outer_position) [0] 
    let inner_position = inner
    let inner_class = replicate (length inner_position) [1]
    let outer_object = zip outer_position outer_class :: [([Double], [Double])]
    let inner_object = zip inner_position inner_class :: [([Double], [Double])]
    let output = outer_object ++ inner_object
    return $ output


getHits :: Network -> [([Double], [Double])] -> Double
getHits network [] = 0
getHits network (y:ys) = result where
    (input, output) = y :: ([Double], [Double])
    pred' = (prediction (feedforward input network)) !! 0:: Double
    pred = if pred' > 0.5 then 1 else 0
    hit = if pred == (output !! 0) then 1 else 0
    rest = getHits network ys
    result = hit + rest

getAccuracy :: Network -> [([Double], [Double])] -> Double
getAccuracy network evaluation_set = result where
    denominator = 1 / (fromIntegral . length $ evaluation_set) :: Double
    hits = getHits network evaluation_set
    result = hits * denominator

loop :: Int -> Double -> Network -> [([Double], [Double])] -> IO Network
loop n rate net training_set
    | n < 0 = return net
    | otherwise = do
        let accuracy = getAccuracy net training_set
        printf "Iteration: %03d Accuracy: %.5f\n" n accuracy

        let zgnet = zerograd net
        let gnet = accumulate_grad mse training_set zgnet
        let newnet = updateWeights rate gnet
        newnet `deepseq` return () -- Force evaluate
        loop (n - 1) rate newnet training_set 

plotNetwork :: Double -> Int -> Network -> Matplotlib
plotNetwork border pixels network = plot where
    -- Linear transformation from [0, pixels] to [-border, border]
    matfunc = (\(x, y) -> [2*border*((fromIntegral x)/(fromIntegral pixels)) - border, 2*border*((fromIntegral y)/(fromIntegral pixels)) - border])
    mat = matrix pixels pixels matfunc
    matval = fmap ( (!! 0) . prediction . (\x -> feedforward x network)) mat :: (Matrix Double)
    matval' = toLists (transpose matval)
    plot = mp % (pcolor matval') @@ [o2 "cmap" "RdBu", o2 "vmin" "0", o2 "vmax" "1"] 

plotTrainingSet :: Double -> Int -> [([Double], [Double])] -> Matplotlib
plotTrainingSet border pixels training_set = plot where
    -- Linear transformation from [-border, border] to [0, pixels]
    outer_points = filter (\(x, y) -> y !! 0 == 0) training_set 
    inner_points = filter (\(x, y) -> y !! 0 == 1) training_set
    outer_positions = fmap fst outer_points :: [[Double]]
    inner_positions = fmap fst inner_points :: [[Double]]
    outer_x' = fmap (!! 0) outer_positions :: [Double]
    outer_y' = fmap (!! 1) outer_positions :: [Double]
    inner_x' = fmap (!! 0) inner_positions :: [Double]
    inner_y' = fmap (!! 1) inner_positions :: [Double]
    outer_x = fmap (\x -> (x + border)*(fromIntegral pixels)/(2 * border)) outer_x'
    outer_y = fmap (\x -> (x + border)*(fromIntegral pixels)/(2 * border)) outer_y'
    inner_x = fmap (\x -> (x + border)*(fromIntegral pixels)/(2 * border)) inner_x'
    inner_y = fmap (\x -> (x + border)*(fromIntegral pixels)/(2 * border)) inner_y'
    plot = scatter outer_x outer_y @@ [o2 "color" "#ff001e"] % scatter inner_x inner_y @@ [o2 "color" "#0015ff"] 

main :: IO ()
main = do
  training_set <- make_moon 100 0.1
  model <- inputLayer 2 >>= pushDALayer 3 htan >>= pushDALayer 9 htan >>= pushDALayer 1 htan
  result <- loop 2000 (0.001) model training_set
  
  let border = 2.4
  let pixels = 300
  onscreen $ (plotNetwork border pixels result) % (plotTrainingSet border pixels training_set)
  putStr "Press Enter to exit..."
  hFlush stdout
  _ <- getLine
  return ()
