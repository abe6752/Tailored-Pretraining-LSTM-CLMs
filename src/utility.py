import numpy as np
import pandas as pd
import re
import operator as op
from functools import reduce
from itertools import product
from inspect import stack
import multiprocessing 
#import logging
#from sklearn.externals import joblib
import os
import io
import datetime
import time
import sys
import pickle
import random
from joblib import Parallel, delayed, cpu_count
import logging

def GetTime(return_second=False, return_date=False):
	t = datetime.datetime.fromtimestamp(time.time())
	if return_second:
		return '{}{}{}_{}{}{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
	elif return_date:
		return '{}{}{}'.format(t.year,t.month,t.day)
	else:
		return '{}{}{}_{}{}'.format(t.year, t.month, t.day, t.hour, t.minute)

def search_exist_suffix(f_path):
	dirname, basename = os.path.split(f_path)

	for i in range(1,1000):
		if '.' in basename: # file
			n_name = basename.replace('.', '_{}.'.format(i))
		else:
			n_name = basename + '_' + str(i) # folder
		
		new_name = os.path.join(dirname,n_name)
		if not os.path.exists(new_name):
			return new_name
		# could not find unused filename for 1000 loops
	raise ValueError('cannot find unused folder names')

def MakeFolder(folder_path, allow_override=False, skip_create=False, time_stamp=False):
	"""
	Make a folder
	"""
	today = GetTime(return_date=True)
	if os.path.exists(folder_path) and skip_create:
		if time_stamp:
			return f'{folder_path}_{today}'
		return folder_path

	if os.path.exists(folder_path) and (not allow_override):
		Warning('Specified folder already exists. Create new one')
		folder_path = search_exist_suffix(folder_path)
		
	if time_stamp:
		folder_path = f'{folder_path}_{today}'
		
	os.makedirs(folder_path, exist_ok=allow_override)

	return folder_path

def partitionIntoSubsets(data,sz):
	remainder = data
	while remainder:
		subset = remainder[:sz]
		remainder = remainder[sz:]
		yield subset


def SplitFileToFiles(infname: str, includeheader: bool, nlines_per_file: int, change_input_file_suffix: bool=True, keepinputfile: bool=True):
	# scan the file to count the number of lines
	with open(infname) as fp: 
		if includeheader:
			header = fp.readline()
		
		lines = [line for line in fp] # not sufficient when insufficient memory 
		if len(lines) < nlines_per_file: 
			print('no need to split. Skip the process')
			return 

		# split lines into nlines_per_file
		basename, extension = infname.rsplit('.',1)
		for idx, subset in enumerate(partitionIntoSubsets(lines, nlines_per_file)):
			outfname = f'{basename}_sub{idx}.{extension}'
			with open(outfname, 'w') as of:
				if includeheader:
					of.write(header)
				of.writelines(subset)

	if change_input_file_suffix and keepinputfile:
		os.rename(infname, f'{infname}.backup')    

	if not keepinputfile:
		os.remove(infname)    

