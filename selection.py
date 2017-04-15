import numpy as np
import tensorflow as tf

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
	

	N, W, H, S = masks.shape
	C = left_image.shape[3]
	shift_images = np.zeros([N,W,H,C,S])
	for s in np.arange(S):
		shift_images[:,:,:,:,s] = np.roll(left_image, s-left_shift, axis=1) #width is axis 1
	
	right_image = np.sum(shift_images * masks, axis=4) #mask axis is 4
	return right_image

def select(masks, left_image, left_shift=16):
	'''
	assumes inputs:
		masks, shape N, W, H, S
		left_image, shape N, W, H, C
	returns
		right_image, shape N, W, H, C
	'''

	N, W, H, S = tf.shape(masks)
	layers = []
	padded = tf.pad(left_image, [[0,0],[left_shift, left_shift],[0,0],[0,0]], mode='REFLECT')
	for s in np.arange(S):
		layers.append(tf.multiply(tf.slice(masks, [0,0,0,s], [N,W,H,1]),
			tf.slice(padded, [0,s,0,0], [N,W,H,-1])))
	return layers.add_n(layers)


if __name__ == '__main__':
	image = tf.image.decode_jpeg('demo.jpg',channels=3)
	

