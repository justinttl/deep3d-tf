import tensorflow as tf
import utils
import os
import os.path
import h5py


class Global_Disp_Model:
    """
        A baseline for deep3d, learns a global disparity to generate right frame from left.
        """
    def __init__():
        self.disparity = tf.

    def save_npy(self, sess, npy_path="./global_disparity-save.npy"):
        assert isinstance(sess, tf.Session)
        np.save(npy_path, self.weights)
        print(("file saved", npy_path))
        return npy_path
