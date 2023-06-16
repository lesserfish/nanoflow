module Nanoflow.Gradient where
import System.Random
import Data.Hashable


-- Generates a name so you don't have to manually name your variables.
generateName :: IO String
generateName = do
  randomInt <- randomIO :: IO Int
  return $ show randomInt

hashString :: String -> String
hashString = show . hash

data UnaryOP = UnaryOP {unName :: String, unEvaluate :: Double -> Double, unGrad :: Double -> Double}
data BinaryOP = BinaryOP {biName :: String, biEvaluate :: Double -> Double -> Double, biLGrad :: Double -> Double -> Double, biRGrad :: Double -> Double -> Double}

data Node = Node {nname :: String, nvalue :: Double, ngrad :: Double} deriving Show

data Expression =   Value {enode :: Node} |
                    Arrow {enode :: Node, eargument :: Expression, afunction :: UnaryOP} |
                    Fork {enode :: Node, elhs :: Expression, erhs :: Expression, ffunction :: BinaryOP}

param :: Double -> IO Expression
param value = do
    name <- generateName
    let node = Node name value 0 :: Node
    let expr = Value node
    return $ expr

collapse :: Expression -> Expression
collapse expr = Value (Node (nname . enode $ expr) (nvalue . enode $ expr) (ngrad . enode $ expr))


-- In this scenario we have
-- y = x
-- We know grad_y = grad_x
xbackpropagate :: Expression -> Double -> Expression
xbackpropagate (Value node) grad_y = output where
    node' = Node (nname node) (nvalue node) grad_y
    output = Value node'

-- In this scenario we have
-- y = F(x)
-- We know grad_y = de/dy
-- We calculate grad_x = de/dx = de/dy * dy/dx = de/dy * F'(x)
xbackpropagate (Arrow node argument function) grad_y = output where
    x = evaluate argument
    dFdx = (unGrad function) x
    grad_x = grad_y * dFdx
    
    node' = Node (nname node) (nvalue node) grad_y
    argument' = xbackpropagate argument grad_x
    output = (Arrow node' argument' function)

-- In this scenario we have
-- y = F(x, z)
-- We know grad = grad_y = de/dy
-- We calculate 
-- grad_x = de/dx = de/dy * dy/dx = de/dy * dF/dx
-- grad_z = de/dz = de/dy * dy/dz = de/dy * dF/dz
xbackpropagate (Fork node lhs rhs function) grad_y = output where
    x = evaluate lhs
    z = evaluate rhs
    dFdx = (biLGrad function) x z
    dFdz = (biRGrad function) x z
    grad_x = grad_y * dFdx
    grad_z = grad_y * dFdz

    node' = Node (nname node) (nvalue node) grad_y
    lhs' = xbackpropagate lhs grad_x
    rhs' = xbackpropagate rhs grad_z
    output = (Fork node' lhs' rhs' function)

backpropagate :: Expression -> Expression
backpropagate x = xbackpropagate x 1

evaluate :: Expression -> Double
evaluate = nvalue . enode

-- In this scenario we have
-- y = F(x)
pushArrow :: UnaryOP -> Expression -> Expression
pushArrow function expression = output where
    x = evaluate expression
    y = (unEvaluate function) x
    name = hashString ((unName function) ++ "(" ++ (nname . enode $ expression) ++ ")")
    node = Node name y 0
    output = (Arrow node expression function)

-- In this scenario we have
-- y = F(x, y)
pushFork :: BinaryOP -> Expression -> Expression -> Expression
pushFork function lhs rhs = output where
    x = evaluate lhs
    z = evaluate rhs
    y = (biEvaluate function) x z
    name = hashString ((biName function) ++ "(" ++ (nname . enode $ lhs) ++ "," ++ (nname . enode $ rhs) ++ ")")
    node = Node name y 0
    output = Fork node lhs rhs function

ngradient :: Expression -> String -> Double 
ngradient (Value node) name = if (nname node) == name then (ngrad node) else 0
ngradient (Arrow node expr _) name = if (nname node) == name then (ngrad node) else ngradient expr name
ngradient (Fork node lhs rhs _) name = if (nname node) == name then (ngrad node) else (ngradient lhs name) + (ngradient rhs name)

gradient :: Expression -> Expression -> Double
gradient f x = ngradient f (nname . enode $ x)
    
-- Operations:
--
--
-- (+)(x, y) = x + y
-- d(+)/dx = d(+)/dy = 1
op_sum_evaluate :: Double -> Double -> Double
op_sum_evaluate x y = x + y
op_sum_grad :: Double -> Double -> Double
op_sum_grad _ _ = 1
op_sum = BinaryOP "sum" op_sum_evaluate op_sum_grad op_sum_grad

-- (-)(x, y) = x - y
-- d(+)/dx = 1
-- d(+)/dy = -1
op_sub_evaluate :: Double -> Double -> Double
op_sub_evaluate x y = x - y
op_sub_lgrad :: Double -> Double -> Double
op_sub_lgrad _ _ = 1
op_sub_rgrad :: Double -> Double -> Double
op_sub_rgrad _ _ = -1
op_sub = BinaryOP "sub" op_sub_evaluate op_sub_lgrad op_sub_rgrad

-- (*)(x, y) = x * y
-- d(+)/dx = y
-- d(+)/dy = x
op_mul_evaluate :: Double -> Double -> Double
op_mul_evaluate x y = x * y
op_mul_lgrad :: Double -> Double -> Double
op_mul_lgrad _ y = y
op_mul_rgrad :: Double -> Double -> Double
op_mul_rgrad x _ = x
op_mul = BinaryOP "mul" op_mul_evaluate op_mul_lgrad op_mul_rgrad

-- (/)(x, y) = x / y
-- d(+)/dx = 1 / y
-- d(+)/dy = -x / y**2
op_div_evaluate :: Double -> Double -> Double
op_div_evaluate x y = x / y
op_div_lgrad :: Double -> Double -> Double
op_div_lgrad _ y = 1/y
op_div_rgrad :: Double -> Double -> Double
op_div_rgrad x y = -x/(y ** 2)
op_div = BinaryOP "div" op_div_evaluate op_div_lgrad op_div_rgrad


-- (**)(x, y) = x ** y
-- d(**)/dx = y * (x ** (y - 1))
-- d(+)/dy = (x**y) * log(y)
op_pow_evaluate :: Double -> Double -> Double
op_pow_evaluate x y = x ** y
op_pow_lgrad :: Double -> Double -> Double
op_pow_lgrad x y = y * (x ** (y - 1))
op_pow_rgrad :: Double -> Double -> Double
op_pow_rgrad x y = (x ** y) * log(x)
op_pow = BinaryOP "pow" op_pow_evaluate op_pow_lgrad op_pow_rgrad

-- relu(x) = relu(x)
-- d / dx = 1 if x >= 0 else 0
op_relu_evaluate :: Double -> Double
op_relu_evaluate x = max 0 x
op_relu_grad :: Double -> Double
op_relu_grad x = if x >= 0 then 1 else 0
op_relu = UnaryOP "relu" op_relu_evaluate op_relu_grad
 
-- exp(x) = exp(x)
-- d / dx = exp (x)
op_exp = UnaryOP "exp" exp exp

-- log(x) = log(x)
-- d / dx = 1/x
op_log_gradient :: Double -> Double
op_log_gradient x = 1/x
op_log = UnaryOP "log" log op_log_gradient

-- sin(x) = sin(x)
-- d / dx = cos x
op_sin = UnaryOP "sin" sin cos

-- cos(x) = cos(x)
-- d / dx = - sin x
op_cos_gradient x = -sin(x)
op_cos = UnaryOP "cos" cos op_cos_gradient

-- asin(x) = asin(x)
-- d / dx = 1/sqrt(1 - x**2)
op_asin_gradient x = 1 / sqrt (1 - x**2)
op_asin = UnaryOP "asin" asin op_asin_gradient

-- acos(x) = acos(x)
-- d / dx = -1/ sqrt(1 - x**2)
op_acos_gradient x = -1 / sqrt(1 - x**2)
op_acos = UnaryOP "acos" acos op_acos_gradient

-- atan(x) = cos(x)
-- d / dx = 1/(1 + x**2)
op_atan_gradient x = 1/(1 + x**2)
op_atan = UnaryOP "atan" atan op_atan_gradient

-- sinh(x) = sinh(x)
-- d / dx = cosh
op_sinh = UnaryOP "sinh" sinh cosh

-- cosh(x) = cosh(x)
-- d / dx = sinh
op_cosh = UnaryOP "cosh" cosh sinh

-- asinh(x) = asinh(x)
-- d / dx = 1/sqrt(x**2 + 1)
op_asinh_gradient x = 1/sqrt(x**2 + 1)
op_asinh = UnaryOP "asinh" asinh op_asinh_gradient

-- acosh(x) = acosh(x)
-- d / dx = 1/sqrt(x**2 - 1)
op_acosh_gradient x = 1/sqrt(x ** 2 - 1)
op_acosh = UnaryOP "acosh" cos op_acosh_gradient

-- atanh(x) = atanh(x)
-- d / dx = 1/(1 - x**2)
op_atanh_gradient x = 1/(1 - x**2)
op_atanh = UnaryOP "atanh" atanh op_atanh_gradient

-- abs(x) = abs(x)
-- d / dx = if x > 0 1 else -1
op_abs_gradient x = if x > 1 then 1 else -1
op_abs = UnaryOP "abs" abs op_abs_gradient

-- signum(x) = signum(x)
-- d / dx = 0
op_signum_gradient x = 0
op_signum = UnaryOP "signum" signum op_signum_gradient


-- Instances 

instance Num Expression where
    (+) = pushFork op_sum
    (-) = pushFork op_sub
    (*) = pushFork op_mul
    abs = pushArrow op_abs
    signum = pushArrow op_signum
    fromInteger x = Value (Node "" (fromInteger x) 0)


instance Fractional Expression where
    (/) a b = pushFork op_div a b
    fromRational x = Value (Node "" (fromRational x) 0)

instance Floating Expression where
    (**) a b = pushFork op_pow a b
    exp = pushArrow op_exp
    log = pushArrow op_log
    cos = pushArrow op_cos
    sin = pushArrow op_sin
    cosh = pushArrow op_cosh
    sinh = pushArrow op_sinh
    asin = pushArrow op_asin
    acos = pushArrow op_acos
    atan = pushArrow op_atan
    asinh = pushArrow op_asinh
    acosh = pushArrow op_acosh
    atanh = pushArrow op_atanh
    pi = Value (Node "" pi 0)

relu :: Expression -> Expression
relu = pushArrow op_relu
