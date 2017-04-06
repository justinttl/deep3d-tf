import sys
import shutil

import skimage
import skimage.io
import skimage.transform

import numpy as np
import os, tarfile
import h5py

def scale_img(path, dims = (180, 320)):
	img = skimage.io.imread(path)

	# Scale down color channels
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


def load_inria_frame(dims = (180,320)):
	print "Collecting Inria Dataset Frames"

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

	# Recursively walk down directorie and collect Frames and concat to X and Y
	dir_name = os.getcwd()
	l_ext = ".jpg"
	r_ext = ".right"

	dimensions = (0, dims[0], dims[1], 3)
	X = np.zeros(dimensions)
	Y = np.zeros(dimensions)

	# Calculate Total Frames
	count = 0 
	ds_count = 0
	tot_count = 0 

	for root, dirs, files in os.walk(dir_name, topdown=False):
		for file in files:
			if file.endswith(l_ext): 
				tot_count = tot_count + 2

	print "Collecting " + str(tot_count) +  " frames and downsampling them."
	print "This will take awhile, be patient =D."

	# The actual walk
	for root, dirs, files in os.walk(dir_name, topdown = True):
		for file in files:
			if file.endswith(l_ext): 
				path = os.path.join(root, file)
				l_frame = scale_img(path, dims).reshape((1, dims[0], dims[1], 3))

				try:
					r_frame = scale_img(path+r_ext, dims).reshape((1, dims[0], dims[1], 3))
					X = np.concatenate((X, l_frame), axis = 0)
					Y = np.concatenate((Y, r_frame), axis = 0)
					count = count + 2
				except IOError:	
					print "Missing Right Frame, Ignoring Entry"

				if count % 250 == 0 and count != 0:
					print  "(" + str(count) + "/" + str(tot_count) + ")"

				if count % 1000 == 0 and count != 0:
					write_to_disk(X,Y, "inria_data.h5", str(ds_count))
					ds_count = ds_count + 1
					del X
					del Y
					X = np.zeros(dimensions)
					Y = np.zeros(dimensions)

	write_to_disk(X,Y, "inria_data.h5", str(ds_count))
	return


def write_to_disk(X, Y, f_name, ds_ext):
	print "Writing " + "X_/Y_" + ds_ext + " to " + f_name
	h5f = h5py.File(f_name, 'a')
	h5f.create_dataset('X_'+ds_ext, data=X)
	h5f.create_dataset('Y_'+ds_ext, data=Y)
	h5f.close()
	print "Write Complete"


def main():
	import time
	start = time.time()

	if os.path.exists('inria_data.h5'):
		print "Some inria data already stored in h5 format"
		return

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
