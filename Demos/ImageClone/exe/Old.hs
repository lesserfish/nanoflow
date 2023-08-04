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

fillColor :: (Word8, Word8, Word8) -> Int -> [Word8]
fillColor (r, g, b) count = concat (replicate count [b, g, r, 1])

colcat :: [(a, a, a, a)] -> [a]
colcat [] = []
colcat (t:ts) = this ++ that where
    (r, g, b, a) = t
    this = [b, g, r, a]
    that = colcat ts

matToArray :: Matrix (Word8, Word8, Word8, Word8) -> [Word8]
matToArray mat = colcat . toList . Data.Matrix.transpose $ mat

data SDLContext = SDLContext {sdlRenderer :: Renderer, sdlTexture :: Texture}
main :: IO ()
main = do
  initializeAll
  window <- createWindow "Image Replicant" defaultWindow
  renderer <- createRenderer window (-1) defaultRenderer
  texture <- createTexture renderer RGB888 TextureAccessStreaming (V2 100 100)
  let context = SDLContext renderer texture
  appLoop context
  destroyTexture texture
  destroyWindow window

appLoop :: SDLContext -> IO ()
appLoop context = do
  let renderer = sdlRenderer context
  let texture = sdlTexture context
  events <- pollEvents
  -- Check Windows Close 
  let eventIsQPress event =
        case eventPayload event of
          KeyboardEvent keyboardEvent -> keyboardEventKeyMotion keyboardEvent == Pressed && keysymKeycode (keyboardEventKeysym keyboardEvent) == KeycodeQ
          WindowClosedEvent _ -> True
          _ -> False
      qPressed = any eventIsQPress events

  -- Fill texture with information
  (pixels, _) <- lockTexture texture Nothing
  let tex_mat = matrix 100 100 (\(i, j) -> (round (fromIntegral i * 255 / 100 :: Double), round (fromIntegral j * 255 / 100 :: Double), 0, 255)) :: Matrix (Word8, Word8, Word8, Word8)
  let raw_data = matToArray tex_mat
  let w8pixels = castPtr pixels :: Ptr Word8
  pokeArray w8pixels raw_data
  unlockTexture texture
  -- 
  let rect = Rectangle (P (V2 15 25)) (V2 500 500) :: Rectangle CInt
  SDL.Video.Renderer.clear renderer
  rendererDrawColor renderer $= V4 127 127 127 255
  SDL.Video.Renderer.copy renderer texture Nothing (Just rect)
  present renderer
  unless qPressed (appLoop context)
