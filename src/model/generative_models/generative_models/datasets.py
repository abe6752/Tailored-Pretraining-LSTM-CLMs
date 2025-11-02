import logging
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import Vocabulary, SmilesTokenizer, is_eligible_smiles
from .utils import Smiles2RDKitCanonicalSmiles
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SmilesDataset(Dataset):
	def __init__(self, smi: pd.Series, vocab: Vocabulary, max_length: Optional [int]=None, phase='test', standerdize_smiles: bool=False):
		self.org_smi	= smi
		if standerdize_smiles:
			self.org_smi = self.org_smi.apply(Smiles2RDKitCanonicalSmiles)
			nmol_org = len(self.org_smi)
			self.na_smi = self.org_smi[self.org_smi.isna()]
			self.org_smi.dropna(inplace=True)
			nmol_curr = len(self.org_smi)
			if nmol_org > nmol_curr:
				log.info(f'Remove some of the molecule during canonicalization process: {nmol_org} to {nmol_curr}')
				log.info(f'{self.na_smi.to_list()}')
			
		elibigle_smi = self.org_smi.apply(lambda x: is_eligible_smiles(x, vocab))
		self.non_eligible_smi = self.org_smi[~elibigle_smi]	
		self.org_smi = self.org_smi[elibigle_smi]	
		nwrong = len(self.non_eligible_smi)
		if nwrong > 0:
			log.info(f'Non eligible smiles (not found in vocabulary) were detected: {nwrong} smiles')
			log.info(f'{self.non_eligible_smi.to_list()}')

		self.vocab 		= vocab
		self.tokens 	= self.org_smi.apply(lambda x: SmilesTokenizer.tokenize_smiles(x, with_begin_and_end=True)).to_frame()
		self.tokens 	= self.tokens.rename(columns={self.tokens.columns[0]: 'tokens'})
		self.tokens['length'] = self.tokens['tokens'].apply(len)
		self.phase 		= phase
		
		if (max_length is None) or (max_length < 1):
			self.max_length = self.tokens['length'].max()
		else:
			self.tokens = self.tokens[self.tokens['length'] <= max_length]
			# log info
			log.info(f'Maximum sequence length is pecified {max_length}')
			log.info('Checking the current smiles_length')
			log.info(f'The number of loaded smiles: {len(self.org_smi)}')
			log.info(f'The number of smiles less than P{max_length}: {len(self.tokens)}')
		
		# padding to the max length
		self.tokens['tokens'] = self.tokens['tokens'].apply(lambda x: x+[SmilesTokenizer.pad]*(self.max_length - len(x)))
		
	def __getitem__(self, idx: int):
		tokens 		= self.tokens.iloc[idx] # order is preseraved 
		smi_encoded = torch.tensor(self.vocab.encode(tokens['tokens']), dtype=torch.long)
		return smi_encoded # to see whether this is necessary or not
	
	def __len__(self):
		return len(self.org_smi)





