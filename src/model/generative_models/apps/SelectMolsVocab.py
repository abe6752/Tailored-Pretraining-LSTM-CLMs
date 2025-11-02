import pickle
import numpy as np
import pandas as pd
import sys
import argparse 
from typing import Optional
from util.utility import MakeFolder, GetRootLogger
from ..generative_models.tokenizer import create_vocabulary, is_eligible_smiles, count_moltokens

class SelectMolsVocabApp():
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
		parser.add_argument('--tokens',
					help='Token file (tsv) (token, frequency), which is an output of running "count_tokens.py". Note, the order of columns is important (token, frequency)',
					type=str
					)
		parser.add_argument('--freq-colname',
					help='Frequency column name of the token file (tsv), which is an output of running "count_tokens.py".',
					type=str,
					default='frequency'
					)
		parser.add_argument('--outdir',
					help='Output directory',
					type=str
					)
		parser.add_argument('--token-frequency-thres',
					help='Threshould of frequency for tokens. Only selected tokens are used and stored in vocaburary.',
					default=0,
					type=int
					)
		parser.add_argument('--debug',
					help='Debug mode: selecting random first 10,000 samples for tokenization.',
					action=argparse.BooleanOptionalAction
					)
		
		parse 	= parser.parse_args(argv)
		self.fname 	= parse.data
		self.smicol	= parse.smi_colname 
		self.ftoken = parse.tokens
		self.tkcol	= parse.freq_colname
		self.fthres = parse.token_frequency_thres 
		self.outfd 	= parse.outdir
		self.debug 	= parse.debug

	def main(self):
		self.outfd	= MakeFolder(self.outfd, allow_override=True)
		logger  = GetRootLogger(f'{self.outfd}/token_selection_log.txt')
		
		tokens = pd.read_csv(self.ftoken, sep='\t', index_col=0)
		mols   = pd.read_csv(self.fname, sep='\t')
		logger.info(f'Loaded molecules: {len(mols)}')
		
		# token threshold 
		logger.info(f'curating tokens: frequency threshold: {self.fthres}')
		oktokens = tokens[tokens[self.tkcol]>self.fthres]
		logger.info(f'ok tokens: {len(oktokens)}')
		logger.info(oktokens.index)

		# select the eligible SMILES
		vocab = create_vocabulary(oktokens.index)
		mols['is_eligible'] = mols[self.smicol].apply(lambda x: is_eligible_smiles(x, vocab))
		okmols = mols[mols['is_eligible']]
		# token length after including starting and end tokens
		okmols['ntokens_srt_end'] = okmols['rdkit_smiles'].apply(lambda x: count_moltokens(x, include_begin_and_end=True))
		logger.info(f'eligible smiles: {len(okmols)}')

		# save the eligible data 
		pickle.dump(vocab, open(f'{self.outfd}/vocab.pickle', 'wb'), protocol=pickle.DEFAULT_PROTOCOL)
		okmols.to_csv(f'{self.outfd}/passed_mols.tsv', sep='\t')
		logger.handlers.clear()

if __name__ == '__main__':
	SelectMolsVocabApp().main()