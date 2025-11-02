import pickle
import numpy as np
import pandas as pd
import sys
import argparse 
from typing import Optional

import torch
from util.utility import MakeFolder, GetRootLogger
from chemistry.visualization import WriteDataFrameSmilesToXls
from ..generative_models.sampling import LSTMsampler
from ..generative_models.lstm import VanilaLSTM
from ..generative_models.gpt import VanilaGPT
from ..generative_models.utils import Smiles2RDKitCanonicalSmiles

class SamplingApp():
	def __init__(self, argv: Optional[str]=None):
		if argv is None:
			argv = sys.argv[1:] # commandilne argument
		parser = argparse.ArgumentParser()
		parser.add_argument('--model-structure',
					help='.pickle file containing model structure (architecture) parameter dictionary.',
					type=str
					)
		parser.add_argument('--model',
					help='lstm or gpt?',
					type=str,
					default='lstm'
					)
		parser.add_argument('--model-state',
					help='.pth file containing model states.',
					type=str
					)
		parser.add_argument('--vocab',
					help='.pickle file containing vocabulary of the word.',
					type=str
					)
		parser.add_argument('--n',
					help='Number of strings to be generated',
					type=int,
					default=1000
					)
		parser.add_argument('--random-seed',
					help='Random seed to generate strings.',
					type=int,
					default=42
					)
		parser.add_argument('--outdir',
					help='Output directory',
					type=str
					)
		parser.add_argument('--allow-override',
					help='Output directory',
					action=argparse.BooleanOptionalAction
					)
		parser.add_argument('--write-excel-file',
					help='Save excel file as well.',
					action=argparse.BooleanOptionalAction
					)
		parse 	= parser.parse_args(argv)
		self.m_name = parse.model
		self.m_str 	= parse.model_structure
		self.m_state= parse.model_state
		self.n 		= parse.n
		self.vocab  = parse.vocab
		self.outfd 	= parse.outdir
		self.rseed  = parse.random_seed
		self.overrride = parse.allow_override
		self.excel  = parse.write_excel_file
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	def main(self):
		outfd  = MakeFolder(self.outfd, allow_override=self.overrride)
		params = pickle.load(open(self.m_str, 'rb'))
		if self.m_name == 'lstm':
			model = VanilaLSTM(**params)
		else:
			raise NotImplementedError('Only "lstm" is available.')
		# set parameters 
		model.load_state_dict(torch.load(self.m_state))
		model.to(self.device)
		
		vocab = pickle.load(open(self.vocab, 'rb'))

		# sampling 
		sampler = LSTMsampler(model, vocab, self.device, rseed=self.rseed)
		smiles 	= sampler.probability_sampling(self.n)

		pd_smi = pd.Series(smiles, name='smiles').to_frame()
		outfname = f'{self.outfd}/gen_smiles.tsv'
		pd_smi.to_csv(outfname, sep='\t')
		if self.excel:
			WriteDataFrameSmilesToXls(pd_smi, 'smiles', outfname.replace('.tsv','.xlsx'), retain_smiles_col=True)


if __name__ =='__main__':
	SamplingApp().main()