import Nanoflow
import System.Random
import Control.DeepSeq
import Text.Printf (printf)
import Control.Monad
import System.IO


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

getData :: Network -> [([Double], [Double])] -> String
getData network [] = ""
getData network (r:rs) = result where
    (input, output) = r
    correct = output !! 0
    pred = (prediction (feedforward input network)) !! 0
    x = input !! 0
    y = input !! 1
    line = (show x) ++ "\t" ++ (show y) ++ "\t" ++ (show correct) ++ "\t" ++ (show pred)
    result = line ++ "\n" ++ (getData network rs)

exportData :: String -> Network -> [([Double], [Double])] -> IO()
exportData filename network training_set = do
    let content = getData network training_set
    writeFile filename content

string2D :: [([Double], [Double])] -> String
string2D [] = ""
string2D (r:rs) = result where
    (input, pred) = r
    x = input !! 0
    y = input !! 1
    z = pred !! 0
    line = (show x) ++ "\t" ++ (show y) ++ "\t" ++ (show z)
    result = line ++ "\n" ++ (string2D rs)

export2D :: String -> Network -> IO ()
export2D filename network = do
    let range = [-2, -2 + 0.02 .. 2]
    let map = [[x, y] | x <- range, y <- range] :: [[Double]]
    let mapfunc = prediction . (\x -> feedforward x network) :: [Double] -> [Double]
    let pred = fmap mapfunc map :: [[Double]]
    let z = zip map pred :: [([Double], [Double])]
    let content = string2D z
    writeFile filename content

main :: IO ()
main = do
  training_set <- make_moon 100 0.1
  model <- inputLayer 2 >>= pushLayer 3 htan >>= pushLayer 9 htan >>= pushLayer 1 htan
  result <- loop 3000 (0.001) model training_set
  exportData "predictions.txt" result training_set
  export2D "map2d.txt" result
  
  return ()
