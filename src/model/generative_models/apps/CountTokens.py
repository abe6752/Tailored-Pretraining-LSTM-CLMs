import numpy as np
import pandas as pd
import sys
import argparse 
from typing import Optional
from util.utility import MakeFolder, GetRootLogger
from ..generative_models.tokenizer import create_vocabulary, count_freqency_tokens, is_eligible_smiles, count_moltokens
from ..generative_models.utils import Smiles2RDKitCanonicalSmiles
from chemistry.descriptors import HeavyAtomCount

class CountTokenFreqApp():
	def __init__(self, argv: Optional[str]=None):
		if argv is None:
			argv = sys.argv[1:] # commandilne argument
		parser = argparse.ArgumentParser()
		parser.add_argument('--data',
					help='.tsv file should be specified containing smiles to be pursed.',
					type=str
					)
		parser.add_argument('--smi-colname',
					help='Column name of smiles in the tsv file',
					type=str
					)
		parser.add_argument('--outdir',
					help='Output directory',
					type=str
					)
		parser.add_argument('--heavy-atom-ratio-thres',
					help='Heavy atom ratio threshold for removing big molecules. Only less than the threshold of input sample remains.',
					default=1.0,
					type=float
					)
		parser.add_argument('--debug',
					help='Debug mode: selecting random first 10,000 samples for tokenization.',
					action=argparse.BooleanOptionalAction
					)
		
		parse 	= parser.parse_args(argv)
		self.fname 	= parse.data
		self.smicol	= parse.smi_colname 
		self.outfd 	= parse.outdir
		self.thres  = np.clip(parse.heavy_atom_ratio_thres, 0, 1.0) # cap by 1.0
		self.debug 	= parse.debug


	def main(self):
		# run the script
		mols 	= pd.read_csv(self.fname, sep='\t')
		if self.debug:
			mols = mols.sample(10000, random_state=0)
		
		self.outfd	= MakeFolder(self.outfd, allow_override=True)
		logger 	= GetRootLogger(f'{self.outfd}/tokenize_log.txt', clearlog=True)
		logger.info(f'Loaded molecules: {len(mols)}')
		
		# curateion by RDKit
		rdsmi_col = 'rdkit_smiles'
		mols[rdsmi_col] = mols[self.smicol].apply(Smiles2RDKitCanonicalSmiles)
		ngmols = mols[rdsmi_col].isna().sum()
		logger.info(f'cannnot handeld mols in rdkit: {ngmols}')
		okmols 		= mols[~mols[rdsmi_col].isna()]
		failmols 	= mols[mols[rdsmi_col].isna()] 
		logger.info(f'passed mols: {len(okmols)}')

		# molecular size filter # 99 percentile in the number of heavy atoms 
		okmols['nhatoms'] = okmols[rdsmi_col].apply(HeavyAtomCount)
		maxha, minha = okmols['nhatoms'].max(), okmols['nhatoms'].min()
		logger.info(f'maximum heavy atom number is {maxha}')
		logger.info(f'minimum heavy atom number is {minha}')

		# eliminate the 95 percentile in nha
		thres_nha	= okmols['nhatoms'].quantile(self.thres)
		okmols 	  	= okmols[okmols['nhatoms']<=thres_nha] # include the equal
		exmols 		= okmols[okmols['nhatoms']>thres_nha]
		logger.info(f'the NHA percentile {self.thres}')
		logger.info(f'the NHA threshold {thres_nha}')
		logger.info(f'selected molecules: {len(okmols)}')

		# tokenize molecules
		freq=count_freqency_tokens(okmols[rdsmi_col])
		pd.Series(freq, name='frequency').to_csv(f'{self.outfd}/token_frequency.tsv', sep='\t')
		outputcol = [self.smicol, rdsmi_col, 'nhatoms']
		okmols[outputcol].to_csv(f'{self.outfd}/ok_mols.tsv', sep='\t')
		exmols[outputcol].to_csv(f'{self.outfd}/excluded_mols.tsv', sep='\t')
		failmols.to_csv(f'{self.outfd}/failed_smiles.tsv', sep='\t')
		
		logger.handlers.clear()
		
if __name__ == '__main__':
	CountTokenFreqApp().main()
