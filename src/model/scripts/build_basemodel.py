import sys
import pickle
import pandas as pd
import numpy as np
import torch

from path_config import bf # path configuration is loaded

from generative_models.datasets import SmilesDataset
from generative_models.training import GenerativeModelTrainer
from generative_models.lstm import VanilaLSTM
from generative_models.parameters import *
from generative_models.sampling import LSTMsampler

def finetune_lstmmodel(bf):
	model_fd = f'{bf}/results/from_cluster/vanila_lstm/nalyer3_hidden512/model'
	vocab	 = pickle.load(open(f'{model_fd}/vocabulary.pickle', 'rb'))
	lstm 	 = VanilaLSTM(len(vocab), layernorm=False)
	lstm.load_state_dict(torch.load(f'{model_fd}/last_epoch_model.pth'))
	device 	= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	lstm.to(device)
	sampler = LSTMsampler(lstm, vocab, device)
	smiles  = sampler.probability_sampling(100)
	print(smiles)

def train_lstm_for_parameter_search(vocab, smiles, outfd, argv):
	# fixed parameters
	smi_colname ='rdkit_smiles'
	nworker 	= 1
	extra_args = ['--epochs','30', 
		'--num-workers',str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--validation-ratio','0.1',
		'--exclude-pad-loss','1',
		'--early-stopping-patience','4',
		'--data',smiles,
		'--smi-colname',smi_colname,
		'--vocab', vocab,
		'--outdir', outfd
		]
	inargs 		 = argv[1:] + extra_args
	lstm_trainer = GenerativeModelTrainer(inargs)
	lstm_trainer.main()

def train_gpt_model(vocab, smiles, argv):
	# fixed parameters
	smi_colname ='rdkit_smiles'
	nworker 	= 1
	extra_args = ['--epochs','2', 
		'--batch-size','128', 
		'--num-workers',str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--validation-ratio','0.1',
		'--exclude-pad-loss','1',
		'--early-stopping-patience','4',
		'--data',smiles,
		'--smi-colname',smi_colname,
		'--vocab', vocab 
		]
	inargs 		 = argv[1:] + extra_args
	gpt_trainer = GenerativeModelTrainer(inargs)
	gpt_trainer.main()

def leadlike_retrain(bf, lr, batch):
	modelfd	= f'{bf}/results/vanila_lstm/lr0005_embed256_layer4_hdim512'
	mstr	= f'{modelfd}/model/best_model_structure_epoch16.pickle'
	mstate 	= f'{modelfd}/model/best_model_epoch16.pth'
	vocab	= f'{modelfd}/model/vocabulary.pickle'
	leadsmi = f'{bf}/results/leadlike_smiles/leadlike_okmols355660.tsv'
	outfd 	= f'{bf}/results/lstm_trained_leadlike'

	smi_colname ='rdkit_smiles'
	nworker 	= 30
	args  	= ['--epochs','30', 
		'--num-workers', str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--validation-ratio','0.1',
		'--exclude-pad-loss','1',
		'--early-stopping-patience','4',
		'--data',leadsmi,
		'--smi-colname',smi_colname,
		'--vocab', vocab,
		'--outdir', outfd,
		'--lr', str(lr),
		'--model', 'lstm',
		'--load-model',
		'--model-structure', mstr,
		'--model-state', mstate,
		'--layernorm',
		'--batch-size', str(batch),
		'--case', f'lr{lr}_batch{batch}'
		]
	lstm_trainer = GenerativeModelTrainer(args)
	lstm_trainer.main()

if __name__ == '__main__':
	vocab	= f'{bf}/results/prepared_smiles/vocab.pickle'
	smiles	= f'{bf}/results/prepared_smiles/passed_mols.tsv' 
	outfd 	= f'{bf}/results/vanila_lstm'
	
	if 0:
		finetune_lstmmodel(bf)
	if 0:
		train_lstm_for_parameter_search(vocab, smiles, outfd, sys.argv)
	if 0:
		train_gpt_model(vocab, smiles, sys.argv)
	if 1:
		lr = sys.argv[1]
		batch = sys.argv[2]
		leadlike_retrain(bf, lr, batch)

		