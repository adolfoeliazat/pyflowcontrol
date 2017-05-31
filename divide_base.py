#!/usr/bin/env python3
import cv2
import patternrecognition as pr
import os
import shutil



folderpath = 'bi/jpg/'
dest='bi/base_dividida/'
files = os.listdir(folderpath)

for pic in files: 
	print(pic)
	newFolder=dest+pic[:4]
	if not os.path.exists(newFolder):
		os.makedirs(newFolder)
	shutil.move(folderpath+pic, newFolder)
	
	
		
		
	

