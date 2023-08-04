{-# LANGUAGE OverloadedStrings #-}
module Main where

import SDL
import SDL.Video.Renderer
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.C.Types
import Data.Word
import Data.Matrix
import Control.Monad (unless)
import Codec.Picture
import Data.Ord

type RGBA = (Word8, Word8, Word8, Word8)

data SDLContext = SDLContext {sdlRenderer :: Renderer, sdlTexture :: Texture}
data AppContext = AppContext {pixelData :: Matrix RGBA}

pix2greyscale :: PixelRGB8 -> Word8
pix2greyscale (PixelRGB8 r g b) = grey where
    r' = (fromIntegral r) / 255 :: Double
    g' = (fromIntegral g) / 255 :: Double
    b' = (fromIntegral b) / 255 :: Double
    grey' = 0.299 * r' + 0.587 * g' + 0.114 * b'
    grey = round . (255 *) $ (clamp (0, 1-0) grey') :: Word8

greyscale2pix :: Word8 -> RGBA
greyscale2pix a = (a, a, a, 255)

rgb2mat :: DynamicImage -> (Matrix Word8, Int, Int)
rgb2mat (ImageRGB8 image@(Image w h _)) = (matrix_data, w - 1, h - 1) where
    matrix_data = Data.Matrix.transpose $ Data.Matrix.transpose $ matrix (h - 1) (w - 1) (\(j, i) -> pix2greyscale (pixelAt image i j))
rgb2mat _ = error "Image format not supported!"

png2mat :: String -> IO (Matrix Word8, Int, Int)
png2mat path  = do
    image <- readImage path
    case image of 
        Left txt -> error txt
        Right image' -> return (rgb2mat image')

colcat :: [(a, a, a, a)] -> [a]
colcat [] = []
colcat (t:ts) = this ++ that where
    (r, g, b, a) = t
    this = [b, g, r, a]
    that = colcat ts

matToArray :: Matrix RGBA -> [Word8]
matToArray mat = colcat . toList $ mat

main :: IO ()
main = do
  -- Get Image data
  (image_data, width, height) <- png2mat "/home/lesserfish/Documents/Code/nanoflow/Demos/ImageClone/images/10032.png"
  putStrLn . show $ image_data
  let pixel_data = fmap greyscale2pix image_data 
  let appcontext = AppContext pixel_data

  -- Initialize SDL
  initializeAll
  window <- createWindow "Image Replicant" defaultWindow
  renderer <- createRenderer window (-1) defaultRenderer
  texture <- createTexture renderer RGB888 TextureAccessStreaming (V2 (fromIntegral width) (fromIntegral height))
  let sdlcontext = SDLContext renderer texture

  appLoop sdlcontext appcontext
  destroyTexture texture
  destroyWindow window

image2Texture :: Matrix RGBA -> Texture -> IO ()
image2Texture pixel_data texture = do
   (pixels, _) <- lockTexture texture Nothing
   let raw_data = matToArray pixel_data
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
    let offset = (x - 1) * 4 + ((y - 1) * (fromIntegral width) * 4)
    let cptr = plusPtr w8ptr offset
    pokeArray cptr [b, g, r, a]
    unlockTexture texture
    return ()

paintPixels' :: [((Int, Int), RGBA)] -> Int -> Ptr () -> IO()
paintPixels' [] _ _ = return ()
paintPixels' (this:rest) width pixels = do
    let ((x, y), (r, g, b, a)) = this
    let w8ptr = castPtr pixels :: Ptr Word8
    let offset = (x - 1) * 4 + ((y - 1) * width * 4)
    let cptr = plusPtr w8ptr offset
    pokeArray cptr [b, g, r, a]
    paintPixels' rest width pixels

paintPixels :: [((Int, Int), RGBA)] -> Texture -> IO ()
paintPixels pixdata texture = do
    texinfo <- queryTexture texture
    let width = textureWidth texinfo
    (pixels, _) <- lockTexture texture Nothing
    paintPixels' pixdata (fromIntegral width) pixels
    unlockTexture texture

appLoop :: SDLContext -> AppContext -> IO ()
appLoop sdlcontext appcontext = do
  let renderer = sdlRenderer sdlcontext
  let texture = sdlTexture sdlcontext
  let pixel_data = pixelData appcontext

  events <- pollEvents
  -- Check Windows Close 
  let eventIsQPress event =
        case eventPayload event of
          KeyboardEvent keyboardEvent -> keyboardEventKeyMotion keyboardEvent == Pressed && keysymKeycode (keyboardEventKeysym keyboardEvent) == KeycodeQ
          WindowClosedEvent _ -> True
          _ -> False
      qPressed = any eventIsQPress events

  -- Fill texture with information
  image2Texture pixel_data texture
  paintPixel (7, 10) (255, 0, 0, 255) texture
  paintPixels [((1, 1), (0, 0, 255, 255)), ((3, 3), (0, 0, 255, 255))] texture
  let rect = Rectangle (P (V2 0 9)) (V2 270 270) :: Rectangle CInt
  SDL.Video.Renderer.clear renderer
  rendererDrawColor renderer $= V4 127 127 127 255
  SDL.Video.Renderer.copy renderer texture Nothing (Just rect)
  present renderer
  unless qPressed (appLoop sdlcontext appcontext)
