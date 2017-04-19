import sys
import shutil

import skimage
import skimage.io
import skimage.transform

import numpy as np
import os, tarfile
import h5py

def resize_scale_img(path, dims = (180, 320), scale = False):
	img = skimage.io.imread(path)

	# Scale down color channels
	if scale:
		img = img / 255.0
		assert (0 <= img).all() and (img <= 1.0).all()

	# Resize
	resized_img = skimage.transform.resize(img, dims, mode = "reflect")

	return resized_img

def download_inria():
	import urllib2

	urls = ["http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_segmentation.tar",
			"http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_video_segmentation.tar",
			"http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_persondetection_train.tar",
			"http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_persondetection_test.tar",
			"http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_poseestimation_train.tar",
			"http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_poseestimation_test.tar",
			"http://www.di.ens.fr/willow/research/stereoseg/dataset/inria_stereo_dataset_negatives.tar"]


	for url in urls:
		file_name = url.split('/')[-1]
		u = urllib2.urlopen(url)
		f = open(file_name, 'wb')
		meta = u.info()
		file_size = int(meta.getheaders("Content-Length")[0])
		print "Downloading: %s Bytes: %s" % (file_name, file_size)

		file_size_dl = 0
		block_sz = 8192
		while True:
		    buffer = u.read(block_sz)
		    if not buffer:
		        break

		    file_size_dl += len(buffer)
		    f.write(buffer)
		    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
		    status = status + chr(8)*(len(status)+1)
		    print status,

		f.close()


def extract_tar():
	dir_name = os.getcwd()
	extension = ".tar"

	print "Extracting all .tar directories ..."

	for item in os.listdir(dir_name): 
		if item.endswith(extension):
			file_name = os.path.abspath(item) 
			tar_ref = tarfile.open(file_name)
			tar_ref.extractall() 
			tar_ref.close() 
			os.remove(file_name) 
			print "Untarred: " + item

	#Remove Viz Folders with only generic jpg frames
	print "Removing Viz Folders"
	dir_name = os.getcwd()
	try:
		viz1_path = dir_name+"/inria_stereo_dataset/poseestimation_test/visualization"
		viz2_path = dir_name+"/inria_stereo_dataset/poseestimation_train/visualization"
		shutil.rmtree(viz1_path)
		shutil.rmtree(viz2_path)
	except:
		pass


def load_inria_frame(dims = (180,320), terminate = None, prob = 0.2):
	print "Collecting Inria Dataset Frames"

	# Recursively walk down directorie and collect Frames and concat to X and Y
	cur_dir = os.getcwd()
	l_ext = ".jpg"
	r_ext = ".right"
	out_ext = ".jpeg"

	# Add directories
	if not os.path.exists(os.path.join(cur_dir, "frames")):
		print "Creating Frames Folder... "
		os.makedirs(os.path.join(cur_dir, "frames", "train", "left"))
		os.makedirs(os.path.join(cur_dir, "frames", "train", "right"))
		os.makedirs(os.path.join(cur_dir, "frames", "test", "left"))
		os.makedirs(os.path.join(cur_dir, "frames", "test", "right"))

	# Calculate Total Frames
	count = 0 
	tot_count = 0 

	for root, dirs, files in os.walk(cur_dir, topdown=False):
		for file in files:
			if file.endswith(l_ext): 
				tot_count = tot_count + 2


	terminate_count = tot_count
	if terminate is not None:
		if terminate * 2 > tot_count:
			print "Error: Requested number of pairs larger than total number of pairs avail"
			return
		terminate_count = terminate * 2



	print "Collecting " + str(terminate_count) +  " frames and downsampling them."
	print "This will take awhile, be patient =D."


	# Create seeded mask for determining Test/Train Split
	np.random.seed(100)
	test_mask = np.random.choice([0,1], size =(terminate_count/2,), p = [1-prob, prob])


	# The actual walk
	for root, dirs, files in os.walk(cur_dir, topdown = True):
		for file in files:
			if file.endswith(l_ext): 
				path = os.path.join(root, file)

				try:
					l_frame = resize_scale_img(path, dims)
					r_frame = resize_scale_img(path+r_ext, dims)
					
					# Write back to disk in designated Folder 
					if test_mask[count/2] == 0:
						folder = "train"
					else:
						folder = "test"
						
					l_path = os.path.join(cur_dir, "frames", folder, "left", str(count/2)+out_ext)
					r_path = os.path.join(cur_dir, "frames", folder, "right", str(count/2)+out_ext)
					skimage.io.imsave(l_path, l_frame)
					skimage.io.imsave(r_path, r_frame)
					count = count + 2
				except IOError:	
					print "IO Error (Missing Right Frame/Mem Limit/etc) , Ignoring Entry"

				if count % 250 == 0 and count != 0:
					print  "(" + str(count) + "/" + str(terminate_count) + ")"

				if count == terminate_count:
					break 
		if count == terminate_count:
			break 

	return


# H5 Writing to Disk Function
def write_to_disk(X, Y, f_name, ds_ext):
	print "Writing " + "X_/Y_" + ds_ext + " to " + f_name
	h5f = h5py.File(f_name, 'a')
	h5f.create_dataset('X_'+ds_ext, data=X)
	h5f.create_dataset('Y_'+ds_ext, data=Y)
	h5f.close()
	print "Write Complete"

# -------------------------------------------------------------- #

def main():
	import time
	start = time.time()

	if os.path.exists('inria_stereo_dataset/'):
		print "Inria data already downloaded. Using cached files"
	else: 
		download_inria()
		extract_tar()

	load_inria_frame()

	end = time.time()
	print "Extraction Completed... Took " + str((end - start)/60) + " minutes"



if __name__ == "__main__":
	main()
