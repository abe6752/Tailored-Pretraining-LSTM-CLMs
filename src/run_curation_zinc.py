""""
2022. 11. 18
These codes are for curating entire ZINC database (tranche-wise downloaded from ZINC15)
to upload to database (postgreSQL)

"""

import glob
import pandas as pd
import os
import shutil
import numpy as np
import sys
import time
from pathlib import Path
from dataset_curation import SmilesToWashedSmiles, STANDARD_KEYS_FOR_WASHED_MOLS
from utility import MakeFolder, SplitFileToFiles
from joblib import cpu_count, Parallel, delayed

from openeye.oechem import *


homedir = os.environ['HOME']
OESetLicenseFile(os.path.join(homedir, '.OpenEye', 'oe_license.txt'))
if OEChemIsLicensed():
    print('OpenEye certification succeeded!')
else:
    print('OpenEye certification failed...')

def test_zinc_curateion(fname: str, outfname: str):
	zincmols = pd.read_csv(fname, sep=',', index_col=0)
	zinc_smiles_name = 'zinc_registered_smiles'
	zincmols.rename(columns={'smiles': zinc_smiles_name}, inplace=True)
	zincmols[STANDARD_KEYS_FOR_WASHED_MOLS] = zincmols[zinc_smiles_name].apply(SmilesToWashedSmiles)
	zincmols.to_csv(outfname, sep=',')


def zinc_curateion_flist(flist: str, outfolder: str):
	# for selected files
	OESetLicenseFile(os.path.join(homedir, '.OpenEye', 'oe_license.txt'))
	if OEChemIsLicensed():
		print('OpenEye certification succeeded!')
	else:
		print('OpenEye certification failed...')
	for fname in flist:
		print('processing ', fname)
		basename = os.path.basename(fname).split('.')[0]
		try:
			zincmols = pd.read_csv(fname, sep='\t', index_col=1, dtype={'features':'str'} ) # zinc id should be the record name
		except:
			print('no data loaded. skip the file')
			statsname = f'{outfd}/error_{basename}.tsv'
			with open(statsname, 'w') as of:
				of.write('loaded mols\tpassedmols\n')
				of.write(f'0\t0\n')
			continue
		
		zinc_smiles_name = 'zinc_registered_smiles'
		zincmols.rename(columns={'smiles': zinc_smiles_name}, inplace=True)
		
		# writing the output directly is better (not concatenating here)
		ofname = f'{outfd}/washed_{basename}.tsv'
		with open(ofname, 'w') as of:
			header = '\t'.join(['zinc_id', zinc_smiles_name, 'mwt','logp', 'reactive', 'purchasable', 'tranche_name'] + STANDARD_KEYS_FOR_WASHED_MOLS)
			of.write(f'{header}\n')
			passedmols = 0
			for value in zincmols.itertuples(name=None): # faster than iterrows
				curated_smiles = SmilesToWashedSmiles(value[1], return_list=True)
				if curated_smiles is None:
					continue	
				passedmols +=1
				inf = '\t'.join([str(value[0]), value[1], str(value[3]), str(value[4]), str(value[5]), str(value[6]), str(value[7])] + curated_smiles)
				of.write(f'{inf}\n')

		statsname = f'{outfd}/count_{basename}.tsv'
		with open(statsname, 'w') as of:
			of.write('loaded mols\tpassedmols\n')
			of.write(f'{len(zincmols)}\t{passedmols}\n')


def curatefiles_infolder(foldername: str, outfd: str, debug: bool=True):
	flist 	= glob.glob(f'{foldername}/*.txt')
	np.random.shuffle(flist) # inplace
	njobs 	= cpu_count() -1

	if debug:
		flist = flist[:njobs]
	
	outfd 	= MakeFolder(outfd, allow_override=True)
	calculated_files  	= [os.path.basename(name).split('.')[0].split('_')[-1] for name in glob.glob(f'{outfd}/count_*.tsv')]
	uncalculated_files 	= [fname for fname in flist if os.path.basename(fname).split('.')[0] not in calculated_files]
	print('#Files to be processed', len(uncalculated_files))	
	batch_fds = np.array_split(np.array(uncalculated_files, dtype='object'), njobs)

	print(f'Pass {calculated_files}')
	# parallel computating
	ret  	= Parallel(n_jobs=njobs)([delayed(zinc_curateion_flist)(fdbatch, outfd) for fdbatch in batch_fds])
	print('Done')

def runSplitToSubfiles(foldername):
	flist 	= glob.glob(f'{foldername}/*.txt')
	nmols_perfile = 1000000
	for fname in flist:
		print(f'processing {fname}', end='\r')
		SplitFileToFiles(fname, nlines_per_file=nmols_perfile, includeheader=True, change_input_file_suffix=True, keepinputfile=False) # discard the input file
		
def moveFilesIntoSubs(foldername):
	flist 	= glob.glob(f'{foldername}/*.txt')
	basefd 	= os.path.dirname(foldername)
	np.random.shuffle(flist)
	batches = np.array_split(flist, 3)
	ouputfolders = ['class1', 'class2', 'class3']

	for batch, outdir in zip(batches, ouputfolders):
		outfd = MakeFolder(f'{basefd}/{outdir}', allow_override=True)
		for fname in batch:
			shutil.move(fname, outfd)


if __name__ == '__main__':
	ROOT = Path(__file__).resolve().parent[1]
	zinc_fd = ROOT / 'data' / 'raw_data' / 'zinc'
	outfd 	= zinc_fd / 'curated_zinc'
	curatefiles_infolder(f'{zinc_fd}', outfd, debug=False)