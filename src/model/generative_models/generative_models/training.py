import sys
import argparse
import pickle
import random 
import shutil
import os
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from util.utility import MakeFolder, GetRootLogger
from .lstm import VanilaLSTM
from .datasets import SmilesDataset
from .sampling import LSTMsampler
from .parameters import *
from .gpt import VanilaGPT
from .utils import Smiles2RDKitCanonicalSmiles

from chemistry.visualization import WriteDataFrameSmilesToXls

# no logging because rootlogger is set in a GenerativeModelTrainer class

def extract_modeltype(args):
	if "--model" in args:
		model_type = args[args.index("--model") + 1]
	else:
		if np.any(["--model=" in word for word in args]):
			for keyword in args:
				model_type = re.search("(?<=--model\=).*", keyword)
				if model_type is not None:
					model_type = model_type.group()
					break
		else:
			raise ValueError('--mdoel keyword is not specified. Specify the model type either: lstm or gpt.')
	print(model_type)
	return model_type

class GenerativeModelTrainer():
	def __init__(self, sys_argv=None):
		if sys_argv is None:
			sys_argv = sys.argv[1:]
		
		# setting paramerter to be parsed
		parser = argparse.ArgumentParser()
		set_global_configs_(parser)
		set_training_conditions_(parser)
		
		# manual parcing to identify model type
		self.model_type 	= extract_modeltype(sys_argv)
		if self.model_type == 'lstm':
			set_lstm_conditions_(parser)
		elif self.model_type == 'gpt':
			set_gpt_conditions_(parser) # might be necessary or not
		
 		# command line arguments parsed.
		self.cli_args = parser.parse_args(sys_argv)
		args 		  = self.cli_args # for shortening
		outdir		  = f'{args.outdir}/{args.case}' if args.case !='' else f'{args.outdir}'
		self.outdir	  = MakeFolder(outdir, allow_override=args.override_folder)
		self.log 	  = GetRootLogger(f'{self.outdir}/generation_log_{args.model}.txt', clearlog=True)	
	
		# random seed setting for reproduciability
		self.init_rngs(args.random_seed)

		# other settings 
		self.use_cuda  = torch.cuda.is_available()
		self.device    = torch.device('cuda' if self.use_cuda else 'cpu')
		self.vocab 	   = pickle.load(open(args.vocab, 'rb'))
		self.padid	   = self.vocab.token2id[self.vocab.pad]
		self.model 	   = self.init_model()
		self.optimizer = self.init_optimizer()
		self.scheduler = self.init_scheduler()
		self.es 	   = self.init_earlystopper()
		self.save_snapshots = args.save_snapshot_models.split(',')
		if len(self.save_snapshots) == 0:
			self.save_snapshots = None
		else: 
			self.save_snapshots = [int(i) for i in self.save_snapshots]

		self.metrics   = ['loss', 'nsamples'] # metrics calculated in addition to loss (?)
		self.write_xlsx = args.write_xlsx
		self.log.info('Model initialization is done')
	
	def init_rngs(self, seed):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	def init_earlystopper(self):
		patience 	= self.cli_args.early_stopping_patience
		self.use_es	= 1 if patience > 0 else 0 # early stopping
		if self.use_es:
			es = EarlyStopper(f'{self.outdir}/during_training', patience=patience)
			return es
		return None

	def init_model(self):
		args = self.cli_args
		if args.model == 'lstm':
			if args.load_model:
				model_str   = pickle.load(open(args.model_structure, 'rb'))
				model_state = torch.load(args.model_state)
				model = VanilaLSTM(**model_str)
				model.load_state_dict(model_state)
			else:
				model = VanilaLSTM(ntokens=len(self.vocab),
							embed_size=args.embed_dim,
							hidden_size=args.hidden_dim,
							nlayers=args.nlayers,
							dropout_lstm=args.dropout_ratio,
							layernorm=args.layernorm,
							padding_idx=self.padid if args.exclude_pad_loss else None,
							) # this could be a paramater later...
		elif args.model == 'gpt':
			model = VanilaGPT(ntokens=len(self.vocab),
					nheads=args.nheads,
					nblocks=args.nblocks,
					nembed=args.embed_dim,
					padding_idx=self.padid if args.exclude_pad_loss else None,
					)
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
	
	def init_scheduler(self):
		return None # for a while # if GPT this is necessary (###)

	def init_dataloders(self):
		args = self.cli_args
		# must be rewritten based on projects
		# load the saved data 
		data 	= pd.read_csv(args.data, sep='\t') # not load index purposely (2024. 04 03 changed)
		smis 	= data[args.smi_colname]		
		if args.debug:
			smis = smis.sample(1000, random_state=0)

		val_ratio = args.validation_ratio
		if val_ratio <= 0:
			self.skip_validation = True
			self.log.info('Validation will be skipped.')
			tr_idx = smis.index
		else:
			self.skip_validation = False
			tr_idx, ts_idx = train_test_split(smis.index, 
										test_size=args.validation_ratio, 
										random_state=args.random_seed)
		# prepare the data set
		batch_size = args.batch_size
		if self.use_cuda: 
			batch_size *= torch.cuda.device_count() # each cuda has the batch size specified
		
		tr_ds = SmilesDataset(smis.loc[tr_idx], self.vocab, standerdize_smiles=args.standardize_smiles)
		
		tr_dl = DataLoader(
			tr_ds,
			batch_size=batch_size,
			num_workers=args.num_workers,
			pin_memory=self.use_cuda,
			shuffle=True
		)		

		if not self.skip_validation:
			ts_ds = SmilesDataset(smis.loc[ts_idx], self.vocab, standerdize_smiles=args.standardize_smiles)
			ts_dl = DataLoader(
				ts_ds,
				batch_size=batch_size,
				num_workers=args.num_workers,
				pin_memory=self.use_cuda,
				shuffle=True
			)
		else:
			ts_dl = None # no validation data sets

		return tr_dl, ts_dl
		
	def main(self):
		# initialization 
		self.log.info(f'Starting: {type(self).__name__}, {self.cli_args}')
		args 			= self.cli_args
		metrics_all 	= dict()
		nsampling 		= args.sampling_epoch
		tr_dl, val_dl  	= self.init_dataloders()
		fd_smiles 		= MakeFolder(f'{self.outdir}/sampling', allow_override=True) if nsampling > 0 else None
		snapshot_fd		= MakeFolder(f'{self.outdir}/snapthos', allow_override=True) if self.save_snapshots is not None else None
		
		# do training 
		for eidx in range(1, args.epochs + 1):
			self.log.info('Epoch {}/{}, training: {} and validation:{} batches, size {}*{}'.format(
				eidx,
				args.epochs,
				len(tr_dl),
				len(val_dl) if val_dl is not None else 0,
				args.batch_size,
				(torch.cuda.device_count() if self.use_cuda else 1),
			))

			# training 
			tr_metrics 	= self.train_model(tr_dl, eidx)
			tr_avg_loss = self.calc_average_loss(tr_metrics, 0, nsample_col=1)

			# validation
			if val_dl is not None:
				val_metrics  = self.validate_model(val_dl, eidx)
				val_avg_loss = self.calc_average_loss(val_metrics, 0, nsample_col=1)
				self.log.info('Epoch {} loss: Training {}, Validation {}'.format(eidx,tr_avg_loss,val_avg_loss))
				self.log.info('-'*70)

				if self.use_es:
					should_stop_trining = self.es(val_avg_loss, self.model, eidx, self.use_multi_cuda)
					if should_stop_trining:
						self.log.info('Eearly stopping training: Best epoch: {}, Best validtion loss: {}'.format(
							self.es.best_epoch,
							self.es.min_loss
						))
						break
			else:
				val_avg_loss = None

			# save model 
			if (self.save_snapshots is not None) and (eidx in self.save_snapshots):
				picklename 	= f'epoch{eidx}_model_structure.pickle'
				statename 	= f'epoch{eidx}_model.pth'
				self._save_model(snapshot_fd, picklename, statename)

			# sampling 
			sampling_metrics = self.sampling_smiles(nsampling, eidx, fd_smiles) if nsampling > 0 else {}
			epoch_metrics 	 = {'training_avg_loss': tr_avg_loss, 'validation_avg_loss': val_avg_loss, 'sampling': nsampling} 
			epoch_metrics.update(sampling_metrics)
			
			# stats is stored
			metrics_all[eidx] = epoch_metrics
			pd.DataFrame.from_dict(metrics_all, orient='index').to_csv(f'{self.outdir}/training_metrics.tsv', sep='\t') # for monitoring

		self.log.info('Done training.')
		pd.DataFrame.from_dict(metrics_all, orient='index').to_csv(f'{self.outdir}/training_metrics.tsv', sep='\t')
		self.save_model_data(tr_dl, val_dl)		


	def sampling_smiles(self, n, epoch_id, fd_smiles):
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
		unique 	   = valid_mols.unique()
		avg_length = np.mean([len(smi) for smi in unique]) # average smiles length

		# save data 
		fname 	 = f'{fd_smiles}/epoch{epoch_id}_{n}samples.tsv'
		pd_smiles.to_csv(fname, sep='\t')

		if self.write_xlsx:
			WriteDataFrameSmilesToXls(pd_smiles, 'canonical_smiles', fname.replace('.tsv', '.xlsx'), retain_smiles_col=True)	

		# return metrics
		return dict(validity=float(nvalid)/len(pd_smiles), 
			  		unique=float(len(unique))/nvalid if nvalid > 0 else 0,
					avg_length=avg_length
					)

	def _save_model(self, folder, struct_name, state_name):
		# struct_name: model structure filename 
		# module is a wrapper class to hundle multiple gpus (this module is removed.)
		model 	= self.model.module if self.use_multi_cuda else self.model 
		p_path 	= os.path.join(folder, struct_name)
		s_path 	= os.path.join(folder, state_name)
		pickle.dump(model.save_data(), open(p_path, 'wb'))
		torch.save(model.to('cpu').state_dict(), s_path)

	def save_model_data(self, tr_dl, val_dl):
		model_folder = MakeFolder(f'{self.outdir}/model', allow_override=True)
		data_folder  = MakeFolder(f'{self.outdir}/datasets', allow_override=True)
		self._save_model(model_folder,'last_epoch_model_structures.pickle', 'last_epoch_model.pth')		
		pickle.dump(self.vocab, open(f'{model_folder}/vocabulary.pickle', 'wb'))
		pickle.dump(tr_dl.dataset, open(f'{data_folder}/training_dataset.pickle','wb'))
		if val_dl is not None:
			pickle.dump(val_dl.dataset, open(f'{data_folder}/validation_dataset.pickle','wb'))
		if self.use_es: # copy best model to the folder
			shutil.copy(self.es.path_to_best_model, f'{model_folder}/best_model_epoch{self.es.best_epoch}.pth')
			shutil.copy(self.es.path_to_best_modelstrs, f'{model_folder}/best_model_structure_epoch{self.es.best_epoch}.pickle')

	def calc_average_loss(self, metrics, avg_col, nsample_col):
		return float(torch.sum(metrics[:,avg_col] * metrics[:,nsample_col])/torch.sum(metrics[:,nsample_col])) # average loss 

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

	def validate_model(self, ts_dl, epoch_idx):
		with torch.no_grad():
			self.model.eval()
			nbatches   = len(ts_dl)
			metrics = torch.zeros(len(ts_dl),len(self.metrics),device=self.device)
			for batch_idx, batch in enumerate(ts_dl):
				loss = self.compute_batch_loss(batch) # metric is necessary?
				self.log.info('Epoch {} -- Batch {}/ {}, validation loss {}'.format(
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


"""
Controlling early stopping options
"""		
class EarlyStopper():
	def __init__(self, tmp_save_folder, patience=5):
		self.patience 	= patience
		self.counter 	= 0 
		self.best_epoch = -1
		self.min_loss 	= float(torch.inf)
		self.tmp_fd 	= MakeFolder(tmp_save_folder, allow_override=True)
		self.path_to_best_model 	= f'{self.tmp_fd}/current_best.pth'
		self.path_to_best_modelstrs	= f'{self.tmp_fd}/current_best_structure.pickle'
	def __call__(self, val_loss, model, epoch, use_multi_cuda):
		if val_loss > self.min_loss:  # 
			self.counter += 1
			if self.counter > self.patience:
				# early stop here 
				return True
		else:
			if use_multi_cuda: 
				model = model.module # module is a wrapper class to hundle multiple gpus (this module is removed.)
			self.counter = 0 # reset counter
			pickle.dump(model.save_data(), open(self.path_to_best_modelstrs, 'wb'))
			torch.save(model.to('cpu').state_dict(),self.path_to_best_model)  # override to save storage
			self.min_loss = val_loss
			self.best_epoch = epoch
			return False




		
		
