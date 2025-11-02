import pickle
import glob
import pandas as pd
import numpy as np
import logging 
import sys 
import torch
import re
from ast import literal_eval
from path_config import bf, results_fd
from generative_models.lstm import VanilaLSTM
from generative_models.sampling import LSTMsampler
from generative_models.tokenizer import Vocabulary, SmilesTokenizer
from generative_models.finetune import Finetuner
from util.utility import GetRootLogger, MakeFolder



def finetune_lstmmodel(outfd, argv):
	smi_colname ='scaf'
	nworker 	= 20
	extra_args = ['--epochs','20', 
		'--batch-size','5', 
		'--num-workers',str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--exclude-pad-loss','1',
		'--data','data/mol-scaf-v1.tsv',
		'--smi-colname',smi_colname,
		'--outdir', outfd,
		'--use-cpus'
		]
	inargs 		 = argv[1:] + extra_args
	finetuner = Finetuner(inargs)
	finetuner.main()

	
if __name__ == '__main__':
	smiles	= f'{bf}/data/fine_tuned_mol/passed_mols.tsv' 
	outfd 	= f'{bf}/results/vanila_lstm/fine_tuning'
	
	finetune_lstmmodel(outfd, sys.argv)