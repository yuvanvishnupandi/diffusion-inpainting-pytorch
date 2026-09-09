import numpy as np
from PIL import Image
import cv2

class MaskGenerator:
    def __init__(self, size):
        self.size = size

    def _blank(self):
        return np.zeros((self.size[1], self.size[0]), dtype=np.uint8)

    def get_rect_mask(self):
        # make a random rectangle
        mask = self._blank()
        w, h = self.size
        
        mw = int(np.random.uniform(0.1, 0.4) * w)
        mh = int(np.random.uniform(0.1, 0.4) * h)
        
        x = np.random.randint(0, w - mw)
        y = np.random.randint(0, h - mh)
        
        mask[y:y+mh, x:x+mw] = 255
        return Image.fromarray(mask)

    def get_irreg_mask(self):
        # make irregular scratch lines
        mask = self._blank()
        w, h = self.size
        
        strokes = np.random.randint(1, 10)
        for _ in range(strokes):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            length = np.random.randint(10, 100)
            angle = np.random.uniform(0, 2 * np.pi)
            
            for _ in range(length):
                thick = np.random.randint(5, 20)
                cv2.circle(mask, (int(x), int(y)), thick, 255, -1)
                
                angle += np.random.uniform(-0.5, 0.5)
                x += np.cos(angle) * 5
                y += np.sin(angle) * 5
                
                if x < 0 or x >= w or y < 0 or y >= h:
                    break
                    
        return Image.fromarray(mask)

    def get_multi_mask(self):
        # combine different masks
        mask = self._blank()
        n = np.random.randint(2, 5)
        
        for _ in range(n):
            if np.random.rand() > 0.5:
                sub = np.array(self.get_rect_mask())
            else:
                sub = np.array(self.get_irreg_mask())
            mask = np.maximum(mask, sub)
            
        return Image.fromarray(mask)

