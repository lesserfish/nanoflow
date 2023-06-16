module Main where

import Test.Hspec
import GradientTest
import NNTest

main :: IO() 
main = hspec $ do
    GradientTest.test
    NNTest.test
