# Data Pull
This folder contains scripts to pull and pre-process data. Beware, some scripts will take a very long time to run

## To run
python get_inria3D.py 
 
## Dependencies needed
skimage, h5py

Simply install via pip
 
## Output of scripts
The output X,Y files will be written to disk
in h5py numpy arrays files. It is advised to 
load local copies instead of re-downloading and processing. 

__To Extract__

```python
h5f = h5py.File('filename.h5','r')
X = h5f['X'][:]
Y = h5f['Y'][:]
h5f.close()
```

