# SPI-GAN-for-Single-Pixel-Camera
Implementation of SPI-GAN: Towards Single-Pixel Imaging through Generative Adversarial Network

1. First download the STL10 and UCF101 datasets. You can find both of these datasets very easily. 
	
		 If you Want to Create the images that will be fed to the GAN 
		 
2. Run Matlab code "L2Norm_Solution.m" for generating the l2-norm solution. Make Necessary Folders before run. I will also upload the python version of this in future.  
		
		
3. Run "save_numpy.py" to create the .npy file under different settings. 

4. Run "Main_Reconstruction.py" to perform the Training.



### For UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```
Please cite the paper when you use these codes-
	'''@misc{karim2021spigan,
	      title={SPI-GAN: Towards Single-Pixel Imaging through Generative Adversarial Network}, 
	      author={Nazmul Karim and Nazanin Rahnavard},
	      year={2021},
	      eprint={2107.01330},
	      archivePrefix={arXiv},
	      primaryClass={cs.CV}
	}'''
