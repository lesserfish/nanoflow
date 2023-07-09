# Nanoflow

A Haskell statistical framework that is currently in development.
So far, we have two modules which are working. 

## Expression

In expression, you can construct algebraic expressions from other algebraic expressions, and calculate their gradient.

For example (see src-exe/Demos/GradientExample.hs)

    module Demos.GradientExample where
    import Nanoflow.Gradient


    main :: IO()
    main = do
        a <- param 3
        b <- param 6
        let f = exp(a * b - b**2) + 3*sin(a) - cos(b)
        let bp = backpropagate f
        let dfda = gradient bp a -- Outputs -2.969977398421458
        let dfdb = gradient bp b -- Outputs -0.2794156352687436


## NN

We also have a feed forward neural network framework implemented. 

For a working demo, please see *src-exe/Demos/NNMoons.hs* and *src-exe/Demos/NNMoonsPlot.hs*


![Figure_1](https://github.com/lesserfish/nanoflow/assets/73536889/3fa685c7-f2f7-4bae-88eb-e40a27c67890)
![moon-demo2](https://github.com/lesserfish/nanoflow/assets/73536889/38af635c-1c3b-4508-befd-95ac37ceadba)
