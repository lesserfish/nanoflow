module Main (main) where
import Test.Hspec
import Nanoflow
import Data.Matrix

main :: IO ()
main = hspec $ do
    describe "Backpropagation" $ do
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
            network <- inputLayer 1 >>= pushLayer 1 htan  -- Generate Network
            let m = getElem 1 1(justValue . weights . ntail $ network) -- Get m
            let b = getElem 1 1(justValue . biases  . ntail $ network) -- Get b
            let x = 3 -- Set x
            let c = 5 -- Set c
            let ff = feedforward [x] network  -- Feedforward
            let bp = backpropagate mse [c] ff -- Backpropagation
            let e = (tanh(m*x + b) - c)**2    -- Calculate correct error
            let n1 = tanh(m*x + b)            -- Calculate correct n1
            let n2 = m*x + b                  -- Calculate correct n2
            let dedn1 = 2*(n1 - c)            -- Calculate correct de/dn1
            let dedn2 = dedn1 * (1 / (cosh n2)**2) -- Calculate correct dn/dn2
            let dedm = dedn2 * x              -- Calculate correct de/dm
            let dedb = dedn2                  -- Calculate correct de/db
            -- Compare values with what we got in backpropagation
            let ededn1 = getElem 1 1 (justGrad . nodes $ bp)
            let ededn2 = getElem 1 1 (justGrad . nodes . ntail $ bp)
            let ededm  = getElem 1 1 (justGrad . weights . ntail $ bp)
            let ededb  = getElem 1 1 (justGrad . biases . ntail $ bp)

            ededn1 `shouldBe` dedn1
            ededn2 `shouldBe` dedn2
            ededm `shouldBe` dedm
            ededb `shouldBe` dedb

            return ()
