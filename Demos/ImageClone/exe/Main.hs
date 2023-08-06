{-# LANGUAGE OverloadedStrings #-}
module Main where

import Nanoflow.NN
import SDL
import SDL.Video.Renderer
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.C.Types
import Data.Word
import Data.Matrix
import Data.List
import System.Random
import Text.Printf (printf)
import Control.Monad (unless)
import Control.DeepSeq
import Codec.Picture
import Data.Ord
import Control.Concurrent
import Control.Concurrent.STM

type RGBA = (Word8, Word8, Word8, Word8)
type Sample = ([Double], Double)
type PixelData = ((Int, Int), RGBA)

data SDLContext = SDLContext {sdlRenderer :: Renderer, sdlTexture :: Texture}
data AppContext = AppContext {pixelMatrix :: Matrix RGBA, pixelData :: [PixelData], appNetwork :: Network, trainingData :: [Sample]}
data SharedContext = SharedContext {
                sharedTexture :: Texture, 
                learningRate :: TVar Double, 
                shouldExit :: TVar Bool, 
                textureSize :: (Int, Int)}

word2prob :: Word8 -> Double
word2prob w = (fromIntegral w) / 255

prob2word :: Double -> Word8
prob2word d = round . (255 *) . (clamp (0, 1)) $ d

pixrgb82pix :: PixelRGB8 -> RGBA
pixrgb82pix (PixelRGB8 r g b) = (r, g, b, 255)

greyscale2pix :: Word8 -> RGBA
greyscale2pix a = (a, a, a, 255)

pix2greyscale :: RGBA -> Word8
pix2greyscale (r, g, b, _) = grey where
    r' = (fromIntegral r) / 255 :: Double
    g' = (fromIntegral g) / 255 :: Double
    b' = (fromIntegral b) / 255 :: Double
    grey' = 0.299 * r' + 0.587 * g' + 0.114 * b'
    grey = round . (255 *) $ (clamp (0, 1-0) grey') :: Word8


rgb2mat :: DynamicImage -> (Matrix RGBA, Int, Int)
rgb2mat (ImageRGB8 image@(Image w h _)) = (matrix_data, w - 1, h - 1) where
    matrix_data = Data.Matrix.transpose $ Data.Matrix.transpose $ matrix (h - 1) (w - 1) (\(j, i) -> pixrgb82pix $ pixelAt image i j) :: Matrix RGBA
rgb2mat _ = error "Image format not supported!"

png2mat :: String -> IO (Matrix RGBA, Int, Int)
png2mat path  = do
    image <- readImage path
    case image of 
        Left txt -> error txt
        Right image' -> return (rgb2mat image')

pixmat2pixdata :: Matrix RGBA -> [PixelData]
pixmat2pixdata mat = output where
    nrow = Data.Matrix.nrows mat
    ncol = Data.Matrix.ncols mat
    helper_mat = matrix nrow ncol (\(i,j) -> ((i, j), (getElem i j mat)))
    output = toList $ helper_mat

pix2training :: (Int, Int) -> PixelData -> Sample
pix2training (width, height) ((x, y), color) = (left, right) where
    left = [x', y']
    right = color' 
    color' = (fromIntegral . pix2greyscale $ color) / 255 :: Double
    x' = (fromIntegral x) / (fromIntegral width) :: Double
    y' = (fromIntegral y) / (fromIntegral height) :: Double

pixdata2trainingdata :: (Int, Int) -> [PixelData] -> [Sample]
pixdata2trainingdata ar pixel_data = fmap (pix2training ar) pixel_data

training2pix :: (Int, Int) -> Sample -> PixelData
training2pix (width, height) (left_arr, right)= ((x, y), color) where
    x' = clamp (0, 1) (left_arr !! 0)
    y' = clamp (0, 1) (left_arr !! 1)
    x = round . ((fromIntegral width) *) $ x' :: Int
    y = round . ((fromIntegral height) *) $ y' :: Int
    color' = round . (255 *) . (clamp (0, 1)) $ right :: Word8
    color = greyscale2pix color'

trainingdata2pixdata :: (Int, Int) -> [Sample] -> [PixelData]
trainingdata2pixdata ar training_data = fmap (training2pix ar) training_data

colcat :: [(a, a, a, a)] -> [a]
colcat [] = []
colcat (t:ts) = this ++ that where
    (r, g, b, a) = t
    this = [b, g, r, a]
    that = colcat ts

matToArray :: Matrix RGBA -> [Word8]
matToArray mat = colcat . toList $ mat

image2Texture :: Matrix RGBA -> Texture -> IO ()
image2Texture pixel_matrix texture = do
   (pixels, _) <- lockTexture texture Nothing
   let raw_data = matToArray pixel_matrix
   let w8pixels = castPtr pixels :: Ptr Word8
   pokeArray w8pixels raw_data
   unlockTexture texture
   return ()

paintPixel :: (Int, Int) -> RGBA -> Texture -> IO ()
paintPixel (x, y) (r, g, b, a) texture = do
    texinfo <- queryTexture texture
    let width = textureWidth texinfo
    (pixels, _) <- lockTexture texture Nothing
    let w8ptr = castPtr pixels :: Ptr Word8
    let offset = (y - 1) * 4 + ((x - 1) * (fromIntegral width) * 4)
    let cptr = plusPtr w8ptr offset
    pokeArray cptr [b, g, r, a]
    unlockTexture texture
    return ()

paintPixels' :: [((Int, Int), RGBA)] -> Int -> Ptr () -> IO()
paintPixels' [] _ _ = return ()
paintPixels' (this:rest) width pixels = do
    let ((x, y), (r, g, b, a)) = this
    let w8ptr = castPtr pixels :: Ptr Word8
    let offset = (y - 1) * 4 + ((x - 1) * width * 4)
    let cptr = plusPtr w8ptr offset
    pokeArray cptr [b, g, r, a]
    paintPixels' rest width pixels

paintPixels :: [((Int, Int), RGBA)] -> Texture -> IO ()
paintPixels pixel_matrix texture = do
    texinfo <- queryTexture texture
    let width = textureWidth texinfo
    (pixels, _) <- lockTexture texture Nothing
    paintPixels' pixel_matrix (fromIntegral width) pixels
    unlockTexture texture

net2pixdata :: Network -> (Int, Int) -> [PixelData]
net2pixdata network (width, height) = pixel_data where
    pixel_data = toList helper_mat
    helper_mat = matrix width height (\(i, j) -> ((i, j), id2val i j)) :: Matrix ((Int, Int), RGBA)
    id2val i j = greyscale2pix . prob2word . (!! 0) . prediction . (\x -> feedforward x network) $ [i', j'] where
        i' = word2prob . fromIntegral $ i
        j' = word2prob . fromIntegral $ j

adjust_output :: Activator
adjust_output = Activator (fmap ((0.5 *) . (0.5 *))) (fmap (0.5 *)) "adjust_output"

sample2String :: Sample -> String
sample2String (left, right) = left' ++ " - " ++ right' where
    left' = "[" ++ (printf "%.3f" (left !! 0)) ++ ","++ (printf "%.3f" (left !! 1)) ++ "]" :: String
    right' = printf "%.3f" right :: String

trainingdata2string :: [Sample] -> String
trainingdata2string t = result where
    result = concat . intersperse "\n" . (fmap show) . (fmap sample2String) $ t where

pixdata2string :: [PixelData] -> String
pixdata2string [] = ""
pixdata2string (y:ys) = this ++ "\n" ++ that where
    ((w, h), color) = y
    this = (printf "%3d" w) ++ " - "++ (printf "%3d" h) ++ " : " ++ (printf "%.3d" (pix2greyscale color))
    that = pixdata2string ys

sample :: Double -> [a] -> IO [a]
sample _ [] = return []
sample p (y:ys) = do
    that <- sample p ys
    chance <- randomRIO (0, 1)
    let this = if chance < p then [y] else []
    return (this ++ that)
    

trainNetwork :: Int -> Double -> Network -> [Sample] -> IO Network
trainNetwork n rate net training_set
    | n < 0 = return net
    | otherwise = do
        training_sample <- sample 0.3 training_set
        let ratio = 1 / (fromIntegral . length $ training_sample) :: Double
        let zgnet = zerograd net
        let gnet = accumulate_grad mse (fmap (\(x, y) -> (x, [y])) training_sample) zgnet
        let newnet = updateWeights (ratio * rate) gnet
        newnet `deepseq` return () -- Force evaluate
        trainNetwork (n - 1) rate newnet training_set 

main :: IO ()
main = do
  -- Get Image data
  (image_data, width, height) <- png2mat "/home/lesserfish/Documents/Code/nanoflow/Demos/ImageClone/images/10033.png"
  let pixel_data = pixmat2pixdata image_data
  let training_data = pixdata2trainingdata (width, height) pixel_data
  putStrLn . show . (fmap pix2greyscale) $ image_data
  putStrLn $ trainingdata2string training_data

  -- Initialize SDL
  initializeAll
  window <- createWindow "Image Replicant" defaultWindow
  renderer <- createRenderer window (-1) defaultRenderer
  sharedtexture <- createTexture renderer RGB888 TextureAccessStreaming (V2 (fromIntegral width) (fromIntegral height))
  sdltexture <- createTexture renderer RGB888 TextureAccessStreaming (V2 (fromIntegral width) (fromIntegral height))
  paintPixels pixel_data sdltexture
  
  let sdlcontext = SDLContext renderer sdltexture

  -- Initialize Neural Network

  network <- inputLayer 2 >>= pushDALayer 64 ((-1.4), 1.4) sigmoid >>= pushDALayer 1 ((-1.4), 1.4) sigmoid
  let appcontext = AppContext image_data pixel_data network training_data 

  learning_rate <- newTVarIO 1.0
  should_exit <- newTVarIO False

  let sharedcontext = SharedContext sharedtexture learning_rate should_exit (width, height)
  toQuit <- newEmptyMVar

  _ <- forkIO $ (sdlLoop sdlcontext sharedcontext) toQuit
  _ <- forkIO $ (nnLoop appcontext sharedcontext)


  takeMVar toQuit
  destroyTexture sdltexture
  destroyTexture sharedtexture
  destroyWindow window

getError :: Network -> [Sample] -> Double
getError _ [] = 0
getError network (y:ys) = result where
    (input, output) = y :: ([Double], Double)
    npred = (prediction (feedforward input network)) :: [Double]
    mistake = fmse npred [output] :: Double
    rest = getError network ys
    result = mistake + rest

cost :: Network -> [Sample] -> Double
cost network training_data = result where
    denominator = 1 / (fromIntegral . length $ training_data) :: Double
    hits = getError network training_data
    result = hits * denominator

nmPress :: Keycode -> Double
nmPress input
    | input == KeycodeI = (-1)
    | input == KeycodeO = 1
    | input == KeycodeU = (-10)
    | input == KeycodeP = 10
    | otherwise = 0

nnLoop :: AppContext -> SharedContext -> IO ()
nnLoop appcontext sharedcontext = do
    let network = appNetwork appcontext
    let training_data = trainingData appcontext
    let (width, height) = textureSize sharedcontext
    let texture = sharedTexture sharedcontext

    learning_rate <- atomically $ readTVar (learningRate sharedcontext) :: IO Double
    should_exit <- atomically $ readTVar (shouldExit sharedcontext) :: IO Bool
    -- Train Network for N epochs
    updated_network <- trainNetwork 1 learning_rate network training_data
    let accuracy = cost network training_data
    printf "Learning Rate: %.3f - Error: %.5f\n" learning_rate accuracy
    let pixel_estimate = net2pixdata network (width, height)
    paintPixels pixel_estimate texture

    let appcontext' = AppContext (pixelMatrix appcontext) (pixelData appcontext) updated_network (trainingData appcontext)
    unless should_exit (nnLoop appcontext' sharedcontext)

sdlLoop :: SDLContext -> SharedContext -> MVar () -> IO ()
sdlLoop sdlcontext sharedcontext toQuit = do
  
  let renderer = sdlRenderer sdlcontext
  let sharedtexture = sharedTexture sharedcontext
  let sdltexture = sdlTexture sdlcontext
  learning_rate <- atomically $ readTVar (learningRate sharedcontext) :: IO Double

  events <- pollEvents
  -- Check Windows Close 
  let eventIsQPress event =
        case eventPayload event of
          KeyboardEvent keyboardEvent -> keyboardEventKeyMotion keyboardEvent == Pressed && keysymKeycode (keyboardEventKeysym keyboardEvent) == KeycodeQ
          WindowClosedEvent _ -> True
          _ -> False
      qPressed = any eventIsQPress events

  let keyHandle event =
        case eventPayload event of
            KeyboardEvent keyboardEvent -> if keyboardEventKeyMotion keyboardEvent == Pressed then nmPress (keysymKeycode (keyboardEventKeysym keyboardEvent)) else 0
            _ -> 0
  let learning_deviation = sum (fmap keyHandle events)
  let learning_rate' = learning_rate + 0.01 * learning_deviation

  let first_rect = Rectangle (P (V2 0 100)) (V2 400 400) :: Rectangle CInt
  let second_rect = Rectangle (P (V2 400 100)) (V2 400 400) :: Rectangle CInt
  SDL.Video.Renderer.clear renderer
  rendererDrawColor renderer $= V4 127 127 127 255
  SDL.Video.Renderer.copy renderer sharedtexture Nothing (Just second_rect)
  SDL.Video.Renderer.copy renderer sdltexture Nothing (Just first_rect)
  present renderer

  atomically $ writeTVar (learningRate sharedcontext) learning_rate'
  atomically $ writeTVar (shouldExit sharedcontext) qPressed

  if qPressed then (putMVar toQuit ()) else return ()
  unless qPressed (sdlLoop sdlcontext sharedcontext toQuit)
