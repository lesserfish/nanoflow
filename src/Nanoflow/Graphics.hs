module Nanoflow.Graphics where
import Nanoflow.NN
import Data.Matrix
import Graphics.Matplotlib

data Node = Node {nposition :: (Double, Double), nvalue :: Double} deriving Show
data Arrow = Arrow {aorigin :: Node, atarget :: Node, avalue :: Double} deriving Show


layerNodes :: Double -> Matrix Parameter -> [Node]
layerNodes x_position nodes = output where
    node_values = toList . justValue $ nodes :: [Double]
    node_length = fromIntegral . length $ node_values
    node_y_positions = fmap (/(max 1 (node_length-1))) [0..(node_length-1)] :: [Double]
    node_x_positions = replicate (length node_values) x_position
    node_positions = zip node_x_positions node_y_positions :: [(Double, Double)]
    output = zipWith Node node_positions node_values

netNodes :: Double -> Network -> [Node]
netNodes x_position  (InputLayer nodes) = layerNodes x_position nodes where
netNodes x_position  (DenseLayer nodes _ _ tail) = output where
    this = layerNodes x_position nodes
    rest = netNodes (x_position - 1) tail
    output = this ++ rest
netNodes x_position (ActivationLayer _ _ tail) = netNodes (x_position) tail

layerArrows :: Double -> Matrix Parameter -> Matrix Parameter -> Network -> [Arrow]
layerArrows x_position nodes weights tail = output where
    previous_nodes = (layerNodes (x_position - 1)) . lnodes $ tail -- has n-elements
    current_nodes = (layerNodes x_position) nodes -- has m elements
    weight_mvalues = justValue weights -- Matrix of size m x n
    nrow = nrows weight_mvalues
    ncol = ncols weight_mvalues
    arrow_matrix = matrix nrow ncol (\(i,j) -> Arrow (previous_nodes !! (j - 1)) (current_nodes !! (i-1)) (getElem i j weight_mvalues)) :: Matrix Arrow
    output = toList arrow_matrix :: [Arrow]

netArrows :: Double -> Network -> [Arrow]
netArrows x_position (InputLayer _) = []
netArrows x_position (DenseLayer nodes weights _ tail) = output where
    this = layerArrows x_position nodes weights tail
    rest = netArrows (x_position - 1) tail
    output = this ++ rest
netArrows x_position (ActivationLayer _ _ tail) = netArrows x_position tail

netGraphics :: Network -> ([Node], [Arrow])
netGraphics network = (nodes, edges) where
    nodes = netNodes 0 network
    edges = netArrows 0 network

plotNodes :: [Node] -> Matplotlib
plotNodes nodes = output where
    x_points = fmap (fst . nposition) nodes
    y_points = fmap (snd . nposition) nodes
    colors = fmap nvalue nodes
    thickness = 140 :: Double
    output = scatter x_points y_points @@ [o2 "c" colors, o2 "s" thickness]

plotArrows :: [Arrow] -> Matplotlib
plotArrows [] = mp
plotArrows (y:ys) = output where
   x0 = fst . nposition . aorigin $ y
   x1 = fst . nposition . atarget $ y
   y0 = snd . nposition . aorigin $ y
   y1 = snd . nposition . atarget $ y
   this = line [x0, x1] [y0, y1]
   rest = plotArrows ys
   output = this % rest

plotGraphics :: ([Node], [Arrow]) -> IO()
plotGraphics (nodes, arrows) = do
    onscreen $ plotNodes nodes % plotArrows arrows

plotNetwork :: Network -> IO()
plotNetwork = plotGraphics . netGraphics
