import sys
import pandas as pd
import pickle
import torch
import glob
import path_config
from util.utility import MakeFolder
from generative_models.lstm import VanilaLSTM
from generative_models.sampling import LSTMsampler
from generative_models.utils import Smiles2RDKitCanonicalSmiles

def sampling_smiles(model_fd, outputfd):
	outfd 		= MakeFolder(outputfd, allow_override=True)
	model_str 	= glob.glob(f'{model_fd}/model/best_model_structure*.pickle')[0]
	model_state = glob.glob(f'{model_fd}/model/best_model*.pth')[0]
	vocab		= pickle.load(open(f'{model_fd}/model/vocabulary.pickle', 'rb'))

	model_str 	= pickle.load(open(model_str, 'rb'))
	lstm 	 	= VanilaLSTM(**model_str)
	lstm.load_state_dict(torch.load(model_state))
	device 	= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	lstm.to(device)

	for i in range(3):
		sampler = LSTMsampler(lstm, vocab, device, rseed=10)
		smiles  = sampler.probability_sampling(100)
		genname = 'generated_smiles'
		pd_smi  = pd.Series(smiles, name=genname).to_frame()
		pd_smi['canonical_smiles'] = pd_smi[genname].apply(Smiles2RDKitCanonicalSmiles)
		pd_smi.to_csv(f'{outfd}/{i}_test.tsv', sep='\t')
		


if __name__ == '__main__':
	bf 		  = '/Users/miyao/work/research/generative_models' 
	model_fd  = f'{bf}/results/from_cluster/lr0.0001_batch128'
	output_fd = f'{bf}/results/sampled_smiles_test'
	sampling_smiles(model_fd, output_fd)