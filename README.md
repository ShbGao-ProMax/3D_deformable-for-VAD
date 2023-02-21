#What is it?
* This is the code for A Deformable L-Unet-3D Based Video Anomaly Detection of High Sensitivity

#Biuld
* 1 Cd to code/dcn.
* 2 For Windows users, run cmd make.bat. For Linux users, run bash make.sh. The scripts will build packages automatically and create some folders.
* 3 See test.py for example usage.

#Dataset
* Download the ped2 dataset and the avenue dataset at [Ped2](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz) and [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip)
* Set the config in train.py

#Train
* Run train.py

#Test
* Run evaluate.py

#Trained weights
* Our trained weights for Ped2 and Avenue which get 97.9% and 86.0% will released when the paper online.
