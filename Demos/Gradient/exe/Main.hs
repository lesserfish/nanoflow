module Main where
import Nanoflow.Gradient


main :: IO()
main = do
    a <- param 3
    b <- param 6
    let f = exp(a * b - b**2) + 3*sin(a) - cos(b)
    let bp = backpropagate f
    let dfda = gradient bp a
    let dfdb = gradient bp b
    putStrLn $ "a = " ++ (show . evaluate $ a)
    putStrLn $ "b = " ++ (show . evaluate $ b)
    putStrLn $ "f = exp(a * b - b**2) + 3*sin(a) - cos(b) = " ++ (show . evaluate $ f)
    putStrLn $ "df/da = " ++ show dfda
    putStrLn $ "df/db = " ++ show dfdb

