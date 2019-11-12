from skimage.transform import resize
import numpy as np

X = np.zeros((480,640,3))
X = resize(X, (240, int(240*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )
print(X.shape)