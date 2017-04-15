import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np


def select(masks, left_image, left_shift=16):
    '''
    assumes inputs:
        masks, shape N, H, W, S
        left_image, shape N, H, W, C
    returns
        right_image, shape N, H, W, C
    '''
    _, H, W, S = masks.get_shape().as_list()
    layers = []
    padded = tf.pad(left_image, [[0,0],[0,0],[left_shift, left_shift],[0,0]], mode='REFLECT')
    # padded is the image padded whatever the left_shift variable is on either side
    for s in np.arange(S):
        layers.append(tf.multiply(tf.slice(masks, [0,0,0,s], [-1, H, W, 1]),
            tf.slice(padded, [0,0,s,0], [-1,H,W,-1])))
    return tf.add_n(layers)


# YOU CAN IGNORE THE FOLLOWING:
def compose(masks, left_image, left_shift=16):
    '''
    THIS FUNCTION IS ASSUMING WIDTH IS ON AXIS 1, MASK AXIS IS 3

    Takes disparity masks and applies pixel selection to generate a right frame

    Inputs:
        masks: narray of shape (N, W, H, S)
        left_images: narray of shape (N, W, H, C)
        left_shift: maximum pixel left shift
    Outputs:
        right_images: narray of shape (N, W, H)
    '''


    N, H, W, C = masks.get_shape().as_list()

    C = left_image.shape[3]
    shift_images = np.zeros([N,W,H,C,S])
    for s in np.arange(S):
        shift_images[:,:,:,:,s] = np.roll(left_image, s-left_shift, axis=1) #width is axis 1

    right_image = np.sum(shift_images * masks, axis=4) #mask axis is 4
    return right_image

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    image_contents = tf.read_file('./test_data/tiger.jpeg')
    image = tf.image.decode_jpeg(image_contents,channels=3)
    print image.eval().shape
    
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # resize to 224, 224
    resized_img = skimage.transform.resize(img, (180, 320))
    return resized_img

