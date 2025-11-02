import sys
import argparse
import logging
import pickle
import random

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from util.utility import MakeFolder, GetRootLogger
from .parameters import set_finetune_conditions_, set_global_configs_, set_training_conditions_
from .lstm import VanilaLSTM
from .gpt 	import VanilaGPT
from .datasets import SmilesDataset
from .sampling import LSTMsampler
from .utils import Smiles2RDKitCanonicalSmiles, canonicalize_smiles

from chemistry.visualization import WriteDataFrameSmilesToXls

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Finetuner():
	def __init__(self, sys_argv=None):
		if sys_argv is None:
			sys_argv=sys.argv[1:]
		
		# setting parameters and parse the 
		parser = argparse.ArgumentParser()
		set_global_configs_(parser)
		set_training_conditions_(parser)
		set_finetune_conditions_(parser) # saved models 
		self.cli_args 	= parser.parse_args(sys_argv)
		args 			= self.cli_args
		self.outdf 		= MakeFolder(args.outdir, allow_override=args.override_folder)
		self.log 		= GetRootLogger(f'{self.outdf}/finetuning_log_{args.model}.txt', clearlog=True)

		self.init_rngs(args.random_seed)
		if args.use_cpus:
			self.use_cuda = False
			self.device = torch.device('cpu')
		else:
			self.use_cuda  = torch.cuda.is_available()
			self.device    = torch.device('cuda' if self.use_cuda else 'cpu')
		self.vocab 	   = pickle.load(open(args.vocab, 'rb'))
		self.padid 		= self.vocab.token2id[self.vocab.pad]
		self.model_type = args.model
		self.model 		= self.init_model()
		self.optimizer  = self.init_optimizer()
		self.es = None
		self.metrics = ['loss', 'nsamples'] # novelty can be calcultated due to small data set for fine tuning

	def init_rngs(self, seed):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	def init_model(self):
		args = self.cli_args
		model_str   = pickle.load(open(args.model_structure, 'rb'))
		model_state = torch.load(args.model_state)
		if args.model == 'lstm': 
			model = VanilaLSTM(**model_str)
			model.load_state_dict(model_state)
		elif args.model == 'gpt':
			model = VanilaGPT(**model_str)
			model.load_stte_dict(model_state)
		else:
			raise NotImplementedError(f'model must be either "lstm" or "gpt".')
		
		if self.use_cuda:
			ngpus = torch.cuda.device_count()
			self.log.info(f'CUDA is on: {ngpus} devices')
			if ngpus > 1:
				model = nn.DataParallel(model)
				self.use_multi_cuda = True
			else:
				self.use_multi_cuda = False
			model = model.to(self.device)
		return model

	def init_optimizer(self):
		return Adam(self.model.parameters(), lr=self.cli_args.lr)
	
	def init_dataloder(self):
		args = self.cli_args
		data = pd.read_csv(args.data, sep='\t')
		smis = data[args.smi_colname]		

		# for finetuning, validation data is not used.
		self.skip_validation = True 
		batch_size = args.batch_size
		if self.use_cuda: 
			batch_size *= torch.cuda.device_count() # each cuda has the batch size specified
		
		tr_ds = SmilesDataset(smis, self.vocab, standerdize_smiles=True) # canonicalize input smiles
		tr_dl = DataLoader(
			tr_ds,
			batch_size=batch_size,
			num_workers=args.num_workers,
			pin_memory=self.use_cuda,
			shuffle=True
		)		
		return tr_dl
			
	def main(self):
		self.log.info(f'Starting: {type(self).__name__}, {self.cli_args}')
		args 			= self.cli_args
		metrics_all 	= dict()
		nsampling 		= args.sampling_epoch
		tr_dl  			= self.init_dataloder()
		fd_smiles 		= MakeFolder(f'{self.outdf}/sampling', allow_override=True) if nsampling > 0 else None
		fd_epoch_models = MakeFolder(f'{self.outdf}/epoch_models', allow_override=True) if args.save_epoch_models else None
		
		# pre-trained model sampling
		pr_metrics 		= self.sampling_smiles(nsampling, 0, fd_smiles, tr_dl.dataset.org_smi)
		metrics_all[0]  = pr_metrics 

		# do training 
		for eidx in range(1, args.epochs + 1):
			self.log.info('Epoch {}/{}, training: {} batches, size {}*{}'.format(
				eidx,
				args.epochs,
				len(tr_dl),
				args.batch_size,
				(torch.cuda.device_count() if self.use_cuda else 1),
			))

			# training 
			tr_metrics 	= self.train_model(tr_dl, eidx)
			tr_avg_loss = self.calc_average_loss(tr_metrics, 0, nsample_col=1)

			# sampling 
			sampling_metrics = self.sampling_smiles(nsampling, eidx, fd_smiles, tr_dl.dataset.org_smi) if nsampling > 0 else {}
			epoch_metrics 	 = {'training_avg_loss': tr_avg_loss, 'sampling': nsampling} 
			epoch_metrics.update(sampling_metrics)
			
			# stats is stored
			metrics_all[eidx] = epoch_metrics
			pd.DataFrame.from_dict(metrics_all, orient='index').to_csv(f'{self.outdf}/finetuning_metrics.tsv', sep='\t') # for monitoring

			# model is also stored (save-epoch-models)
			if fd_epoch_models is not None:
				self.save_model(fd_epoch_models, comment=f'epoch{eidx}')

		self.log.info('Done finetuning.')
		pd.DataFrame.from_dict(metrics_all, orient='index').to_csv(f'{self.outdf}/finetuning_metrics.tsv', sep='\t')
		self.save_model_data(tr_dl)		

	def train_model(self, tr_dl, epoch_idx):
		self.model.train()
		nbatches   = len(tr_dl)
		metrics = torch.zeros(len(tr_dl),len(self.metrics),device=self.device)
		for batch_idx, batch in enumerate(tr_dl):
			self.optimizer.zero_grad()
			loss = self.compute_batch_loss(batch) # metric is necessary?
			loss.backward()
			self.optimizer.step()
			### train_model and validate_model have lots in common. Must be a way to integrate..
			self.log.info('Epoch {} -- Batch {}/ {}, training loss {}'.format(
				epoch_idx,
				batch_idx+1,
				nbatches,
				loss			
			))
			metrics[batch_idx,:] = torch.tensor([loss, len(batch)])
		self.log.info('-'*70)
		return metrics.to('cpu')

	def compute_batch_loss(self, batch):
		x  	= batch[:, :-1]
		y  	= batch[:, 1:] # slide one 
		x  	= x.to(self.device, non_blocking=True)
		y	= y.to(self.device, non_blocking=True)

		predict = self.model(x) # teacher forcing
		loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.padid) #self.padid)
		
		# loss calculation (batch, sequence, token) -> (batch, tokens, sequence)
		loss = loss_func(predict.permute(0,2,1), y)
		return loss

	def calc_average_loss(self, metrics, avg_col, nsample_col): # for epoch
		return float(torch.sum(metrics[:,avg_col] * metrics[:,nsample_col])/torch.sum(metrics[:,nsample_col])) # average loss 


	def sampling_smiles(self, n, epoch_id, fd_smiles, training_smiles):
		if self.model_type == 'lstm':
				sampled_smiles = LSTMsampler(self.model, self.vocab, self.device).probability_sampling(n)
		else:
			raise NotImplementedError('GPT sampling is unimplemented, here.')
		
		pd_smiles = pd.Series(sampled_smiles).to_frame()
		pd_smiles.rename(columns={pd_smiles.columns[0]:'smiles'}, inplace=True)
		
		# Evaluation (simple)
		pd_smiles['canonical_smiles'] = pd_smiles['smiles'].apply(Smiles2RDKitCanonicalSmiles)
		valid_mols = pd_smiles['canonical_smiles'].dropna()
		nvalid     = len(valid_mols)
		unique 	   = valid_mols.drop_duplicates() #modified by Abe
		avg_length = np.mean([len(smi) for smi in unique]) # average smiles length

		# novelty modified by Abe
		train_cano_smi = canonicalize_smiles(training_smiles)
		train_set = set(train_cano_smi)
		isnovel	= ~unique.isin(train_set)
		novelty = isnovel.sum() / len(unique) if len(unique) > 0 else 0.0

		# save data 
		fname 	 = f'{fd_smiles}/epoch{epoch_id}_{n}samples.tsv'
		ngensmiles = len(pd_smiles)
		pd_smiles.to_csv(fname, sep='\t')
		if len(pd_smiles) > 1000:
			logging.info('sampling smiles at most 1000.')
			pd_smiles = pd_smiles.sample(1000, random_state=10)
		fname_xlsx = f'{fd_smiles}/epoch{epoch_id}_{len(pd_smiles)}samples.xlsx'
		WriteDataFrameSmilesToXls(pd_smiles, 'canonical_smiles', fname_xlsx, retain_smiles_col=True)	

		# return metrics
		return dict(validity=float(nvalid)/ngensmiles, 
			  		unique=float(len(unique))/nvalid if nvalid > 0 else 0,
					novelty=novelty,
					avg_length=avg_length
					)
	
	def save_model(self, fd, comment):
		if self.use_cuda and self.use_multi_cuda: 
			model = self.model.module # module is a wrapper class to hundle multiple gpus (this module is removed.)
		else:
			model = self.model
		
		# model architecture
		pickle.dump(model.save_data(), open(f'{fd}/{comment}_model_structures.pickle', 'wb'))
		torch.save(model.to('cpu').state_dict(), f'{fd}/{comment}_model.pth')# model parameters


	def save_model_data(self, tr_dl):
		model_folder = MakeFolder(f'{self.outdf}/model', allow_override=True)
		data_folder  = MakeFolder(f'{self.outdf}/datasets', allow_override=True)
		
		if self.use_cuda and self.use_multi_cuda: 
			model = self.model.module # module is a wrapper class to hundle multiple gpus (this module is removed.)
		else:
			model = self.model
		
		# model architecture
		pickle.dump(model.save_data(), open(f'{model_folder}/last_epoch_model_structures.pickle', 'wb'))
		torch.save(model.to('cpu').state_dict(), f'{model_folder}/last_epoch_model.pth')# model parameters
		
		pickle.dump(self.vocab, open(f'{model_folder}/vocabulary.pickle', 'wb'))
		pickle.dump(tr_dl.dataset, open(f'{data_folder}/training_dataset.pickle','wb'))
		org_smiles = tr_dl.dataset.org_smi
		if isinstance(org_smiles, pd.Series):
			org_smiles 	= org_smiles.to_frame()
			colname 	= org_smiles.columns[0]
			fout 		= f'{data_folder}/smiles_used_for_finetuning.tsv'
			org_smiles.to_csv(fout, sep='\t')
			WriteDataFrameSmilesToXls(org_smiles, colname, fout.replace('.tsv', '.xlsx'))


		
	