# modified from https://github.com/collinswakholi/ImgTiler

import numpy as np
import math

def what_is(image):
    type_ = (str(image.dtype)).lower()
    if 'float' in type_:
        max_val = np.max(image)
        min_val = np.min(image)
    elif 'uint8' in type_:
        max_val = 255
        min_val = 0
    elif 'uint16' in type_:
        max_val = 65535
        min_val = 0
    elif 'uint32' in type_:
        max_val = 4294967295
        min_val = 0
    
    dtype = eval('np.'+type_)
    
    # print("max_val: ", max_val)
    # print("min_val: ", min_val)
    
    return dtype, max_val, min_val

def convert_to_uint8(image, dtype=np.uint8, min_val=0, max_val=255):
    type_ = (str(dtype)).lower()
    if 'uint' in type_:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array((image-min_val)/(max_val-min_val)*255, dtype=np.uint8)
    
    return image

def convert_from_uint8(image, dtype=np.uint8, min_val=0, max_val=255):
    type_ = (str(dtype)).lower()
    if 'float' in type_:
        image = np.array(image/255*(max_val-min_val)+min_val, dtype=dtype)
    elif 'uint8' in type_:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array(round(image/255*(max_val-min_val)+min_val), dtype=dtype)
        
    return image
    
    
class SplitImage:
    """
    Class to split an image into tiles based on a grid (m x n) and overlap between tiles
    """
    def __init__(self, image, grid, overlap):
        """
        Initialize the SplitImage class.

        Args:
            image (ndarray): Input image 
            grid (tuple): Grid dimensions (rows, columns) to split the image.
            overlap (int): Number of pixels to overlap between tiles.
        """
        self.type, self.min_val, self.max_val = what_is(image)
        self.image = convert_to_uint8(image, dtype=self.type, min_val=self.min_val, max_val=self.max_val)
        self.grid = grid
        self.overlap = max(overlap, 1)
        
        # print("\nImage channels: ", image.shape[2] if len(image.shape) == 3 else 1)
        # print("Input data type: ", type)
  
    
    def pad_image(self, image, padding):
        """
        Pad the image with zeros.

        Args:
            image (ndarray): Input image.
            padding (tuple): Padding values for top, bottom, left, and right.

        Returns:
            ndarray: Padded image.
        """
        if len(image.shape) == 2:
            image = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant', constant_values=0)
        else:
            image = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)), 'constant', constant_values=0)

        return image

    def check_divisible(self, image):
        """
        Check if the image dimensions are divisible by the grid dimensions.
        If not, pad the image to make it divisible.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Image with dimensions divisible by the grid.
        """
        h, w = image.shape[:2]
        w_pad = (self.grid[1] - (w % self.grid[1])) % self.grid[1]
        h_pad = (self.grid[0] - (h % self.grid[0])) % self.grid[0]

        new_image = self.pad_image(image, (0, h_pad, 0, w_pad))

        return new_image

    def extract_tile(self, image, tile_height, tile_width, tile_height_pad, tile_width_pad, i, j):
        """
        Extract a tile from the image.

        Args:
            image (ndarray): Input image.
            tile_height (int): Height of the tile.
            tile_width (int): Width of the tile.
            i (int): Row index of the tile.
            j (int): Column index of the tile.

        Returns:
            ndarray: Extracted tile.
        """
        overlap = self.overlap
        x1 = i * tile_height
        x2 = (i + 1) * tile_height + 2 * overlap
        x2 += tile_height_pad

        y1 = j * tile_width
        y2 = (j + 1) * tile_width + 2 * overlap
        y2 += tile_width_pad

        if len(image.shape) == 2:
            tile = image[x1:x2, y1:y2]
        else:
            tile = image[x1:x2, y1:y2, :]
        return tile

    def split_image(self):
        """
        Split the image into multiple tiles.

        Args:
            show_rect (bool, optional): Whether to draw rectangles on the tiles. Defaults to True.
            show_tiles (bool, optional): Whether to display the tiles. Defaults to True.

        Returns:
            list: List of tiles.
        """
        overlap = self.overlap
        image = self.check_divisible(self.image)

        height, width = image.shape[:2]
        tile_height = height // self.grid[0]
        tile_width = width // self.grid[1]
        tile_pad_height = math.ceil((tile_height + 2 * overlap) * (1 / 3))
        tile_pad_width = math.ceil((tile_width + 2 * overlap) * (1 / 3))

        image = self.pad_image(image, (overlap, overlap + tile_pad_height, overlap, overlap + tile_pad_width))
        c = image.shape[2] if len(image.shape) == 3 else 1
        
        tiles = []
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                tile = self.extract_tile(image, tile_height, tile_width, tile_pad_height, tile_pad_width, i, j)

                tile = convert_from_uint8(tile, dtype=self.type, min_val=self.min_val, max_val=self.max_val)
                tiles.append(tile)

        return tiles, tile_pad_height, tile_pad_width

class CombineTiles:
    """
    Class to combine the tiles back into an image.
    """
    def __init__(self, tiles, grid, overlap, tile_pad_height, tile_pad_width):
        """
        Initialize the CombineTiles class.

        Args:
            tiles (list): List of tiles.
            grid (tuple): Grid dimensions (rows, columns) of the tiles.
            overlap (int): Number of pixels overlapped between tiles.
        """
        self.type, self.min_val, self.max_val = what_is(tiles[0])
        self.tiles = [convert_to_uint8(tile, dtype=self.type, min_val=self.min_val, max_val=self.max_val) for tile in tiles]
        self.grid = grid
        self.overlap = max(overlap, 1)
        self.tile_pad_height = tile_pad_height
        self.tile_pad_width = tile_pad_width
        # print("\nTile channels: ", tiles[0].shape[2] if len(tiles[0].shape) == 3 else 1)
        # print("Input data type: ", type)
        
    def depad_tiles(self, tiles):
        """
        Remove the padding from the tiles to get the original tile size.

        Args:
            tiles (list): List of tiles.

        Returns:
            list: List of depadded tiles.
        """
        overlap = self.overlap
        tiles_ = []
        for tile in tiles:
            tiles_.append(tile[overlap:-overlap, overlap:-overlap, :] if len(tile.shape) == 3 else tile[overlap:-overlap, overlap:-overlap])
        return tiles_

    def combine_tiles(self):
        """
        Combine the tiles into an image.
        
        Args:
            show_image (bool, optional): Whether to display the image. Defaults to True.
            
        Returns:
            ndarray: Reconstructed image.
        """
        grid = self.grid
        tiles = self.depad_tiles(self.tiles)

        tile_height, tile_width = tiles[0].shape[:2]
        tile_height -= self.tile_pad_height
        tile_width -= self.tile_pad_width
        image_shape = (tile_height * grid[0], tile_width * grid[1], tiles[0].shape[2]) if len(tiles[0].shape) == 3 else (tile_height * grid[0], tile_width * grid[1])
        image = np.zeros(image_shape, dtype=self.type)

        for i in range(grid[0]):
            for j in range(grid[1]):
                tile = tiles[i * grid[1] + j]
                tile = tile[0:tile_height, 0:tile_width]
                if len(tile.shape) == 3:
                    image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width, :] = tile
                else:
                    image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width] = tile
                   
        image = convert_from_uint8(image, dtype=self.type, min_val=self.min_val, max_val=self.max_val)
        
        return image
