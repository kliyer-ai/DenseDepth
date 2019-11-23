
def get_shape_rgb(batch_size=None):
    dim = 224
    if batch_size: return (batch_size, dim, dim, 3)
    else: return  (dim, dim, 3)

def get_shape_depth(batch_size=None, halved=False):
    dim = 224
    if halved: dim = dim // 2
    if batch_size: return (batch_size, dim, dim, 1)
    else: return  (dim, dim, 1)