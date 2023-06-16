module GradientTest (test) where
import Test.Hspec
import Nanoflow.Gradient


mse :: Double -> Double -> Double
mse a b = sqrt ((a-b)**2)

test :: Spec
test = do
    describe "Forward pass" $ do
        it "checks forward pass of first expression" $ do
            a <- param 4
            b <- param 7
            c <- param 16
            let d = exp (a * b - c ** 2)
            let e = relu (d + (1 / a))
            let f = sin (0.3 * e)
            let output = evaluate f
            let expected = 0.07492970727274234
            let difference = mse output expected
            difference `shouldSatisfy` (< 0.01)
            return()
        return()

    describe "Backpropagation" $ do
        it "checks backpropagation" $ do
            a <- param 1
            b <- param 1
            c <- param 4
            let d = sin (a + b - c ** 2)
            let e = relu (d + (1 / a))
            let f = exp (0.3 * e)
            let x = backpropagate f
            let dxda = gradient x a
            let dxdb = gradient x b
            let dxdc = gradient x c
            let expected_dxda = -0.2597096124673057
            let expected_dxdb = 0.041136917633457656
            let expected_dxdc = -0.32909534106766125
            (mse dxda expected_dxda) `shouldSatisfy` (< 0.01)
            (mse dxdb expected_dxdb) `shouldSatisfy` (< 0.01)
            (mse dxdc expected_dxdc) `shouldSatisfy` (< 0.01)
            return()
    describe "Backpropagation" $ do
        it "checks more backpropagation" $ do
            a <- param 0.3
            b <- param 0.1
            c <- param 1.0
            let d = sinh (a + cosh(c ** 2))
            let e = relu (b + d + sin(a))
            let f = cos ( e - 1/a )
            let g = sin (f ** 3)
            let h = log ( atanh(g) )
            let x = backpropagate h
            let dxda = gradient x a
            let dxdb = gradient x b
            let dxdc = gradient x c
            let expected_dxda = -9.546846434211412
            let expected_dxdb = -0.6238303536604609
            let expected_dxdc = -4.7464747697137355
            (mse dxda expected_dxda) `shouldSatisfy` (< 0.01)
            (mse dxdb expected_dxdb) `shouldSatisfy` (< 0.01)
            (mse dxdc expected_dxdc) `shouldSatisfy` (< 0.01)
            return()
    describe "Backpropagation" $ do
        it "checks even more backpropagation" $ do
            a <- param 0.3
            b <- param 0.5
            let c = asin(a * b)
            let d = cos(c ** 2)
            let e = tanh(d)
            let x = backpropagate e
            let dxda = gradient x a
            let dxdb = gradient x b
            let expected_dxda = -0.0014504327859911142
            let expected_dxdb = -0.0008702597061756809
            (mse dxda expected_dxda) `shouldSatisfy` (< 0.01)
            (mse dxdb expected_dxdb) `shouldSatisfy` (< 0.01)
            return()
    describe "Double backpropagation still works" $ do
        it "checks even more backpropagation" $ do
            a <- param 0.3
            b <- param 0.5
            let c = asin(a * b)
            let d = cos(c ** 2)
            let e = tanh(d)
            let y = backpropagate e
            let x = backpropagate y
            let dxda = gradient x a
            let dxdb = gradient x b
            let expected_dxda = -0.0014504327859911142
            let expected_dxdb = -0.0008702597061756809
            (mse dxda expected_dxda) `shouldSatisfy` (< 0.01)
            (mse dxdb expected_dxdb) `shouldSatisfy` (< 0.01)
            return()

        return()

    return()

