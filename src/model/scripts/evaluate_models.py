import pickle
import glob
import pandas as pd
import numpy as np
import logging 
import sys 
import torch
import re
from joblib import delayed, Parallel
from ast import literal_eval
from itertools import product
from path_config import bf, results_fd
from generative_models.lstm import VanilaLSTM
from generative_models.sampling import LSTMsampler
from generative_models.tokenizer import Vocabulary, SmilesTokenizer
from generative_models.utils import Smiles2RDKitCanonicalSmiles, canonicalize_smiles

from util.utility import GetRootLogger, MakeFolder
from chemistry.visualization import WriteDataFrameSmilesToXls
from chemistry.descriptors import calcRDKitDescriptors_specified, rdkit_descriptorlist
from sklearn.metrics.pairwise import pairwise_distances
from figure.plots import makeHistogramOne, makeHistogramTwo

from rdkit.Chem import AllChem
from rdkit import Chem

# from iwasakikun
def MorganbitCalcAsVector(mol,rad=2,bits=2048,useChirality=True):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol,rad,bits,useChirality=useChirality))

def MorganbitCalcAsVectorFromSmiles(sma,rad=2,bits=2048,useChirality=True):
    return list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sma),rad,bits,useChirality=useChirality))


def get_lstm_parameters(fd_name):
	conditions = {}
	with open(f'{fd_name}/generation_log_lstm.txt', 'r') as fp: 
		lines = fp.readlines()
		for line in lines:
			if 'GenerativeModelTrainer, Namespace' in line: # calculation condition extraction  
				namespace = re.search('(?<=Namespace\().*(?=\))', line).group()
				# parse the conditions 
				for condition in namespace.split(','):
					name,value = condition.split('=')
					value = literal_eval(value)
					conditions[name] = value
	return conditions

def evaluate_lstm_model(fd_name, training_smiles, return_smiles=False, nsamples=10000, add_properties=False, use_last_epoch_model=False):
	model_fd = f'{fd_name}/model'
	vocab	 = pickle.load(open(f'{model_fd}/vocabulary.pickle', 'rb'))
	
	# identifying the best model
	if use_last_epoch_model:
		bm_file 	= f'{model_fd}/last_epoch_model.pth'
		bm_str_file	= f'{model_fd}/last_epoch_model_structures.pickle'
	else:
		bm_file		= glob.glob(f'{model_fd}/best_model_epoch*.pth')[0] # bm: best model
		bm_str_file	= glob.glob(f'{model_fd}/best_model_structure*.pickle')[0]
		bm_epoch  	= int(re.search('(?<=epoch)[0-9]*(?=.pth)',bm_file).group())
	
	bm_state  	= torch.load(bm_file) 
	bm_str 		= pickle.load(open(bm_str_file, 'rb'))
	# loss inforamtion for the best model
	if use_last_epoch_model:
		tr_metrics  = pd.read_csv(f'{fd_name}/training_metrics.tsv', sep='\t', index_col=0).iloc[-1]
	else:
		tr_metrics  = pd.read_csv(f'{fd_name}/training_metrics.tsv', sep='\t', index_col=0).loc[bm_epoch]
	d_metrics 	= tr_metrics.to_dict()
	lstm = VanilaLSTM(**bm_str)
	lstm.load_state_dict(bm_state)
	device 	= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	lstm.to(device)

	sampler = LSTMsampler(lstm, vocab, device)
	smiles  = pd.Series(sampler.probability_sampling(nsamples), name='gen_smiles').to_frame()
	
	# smiles evaluation
	csmi = 'canonical_smiles'
	smiles[csmi] = smiles['gen_smiles'].apply(Smiles2RDKitCanonicalSmiles)
	smiles['isvalid'] = ~smiles[csmi].isna() # save info on the smiles as well 
	validsmis = smiles[smiles['isvalid']]
	
	# uniqueness
	validity = len(validsmis)/len(smiles)
	unique_smiles = validsmis[csmi].drop_duplicates()
	uniqueness = len(unique_smiles) / len(validsmis) if len(validsmis) > 0 else 0

	# novelty
	smiles['isnovel']	 = ~smiles[csmi].isin(training_smiles) 
	train_cano_smi = canonicalize_smiles(training_smiles)
	train_set = set(train_cano_smi)
	isnovel_unique = ~unique_smiles.isin(train_set)
	novelty = isnovel_unique.sum() / len(unique_smiles) if len(unique_smiles) > 0 else 0

	smiles['isnovel'] = False
	mask_valid = smiles['isnvalid']
	smiles.loc[mask_valid, 'isnovel'] = ~smiles.loc[mask_valid, csmi].isin(train_set)

	gen_metrics = {f'validity{nsamples}':validity,f'uniqueness{nsamples}':uniqueness,f'novelty{nsamples}':novelty,'gensmiles':nsamples}
	d_metrics.update(gen_metrics)

	if add_properties:
		props = ['MolWt', 'MolLogP']
		dfprop = calcRDKitDescriptors_specified(smiles[csmi], props, nworkers=10)
		smiles = smiles.join(dfprop)
	if return_smiles:
		return d_metrics, smiles
	else:
		return d_metrics

def evaluate_lstm_parameters(bf, add_props=False):
	outfd	= MakeFolder(f'{bf}/results/analyze_lstmlog', allow_override=True)
	log 	= GetRootLogger(f'{outfd}/stats_log.txt')
	results_fds = glob.glob(f'{results_fd}/vanila_lstm/*/') # this is defined in path_config.py

	
	lstm_parameters= dict()
	for idx, result_fd in enumerate(results_fds):
		log.info(f'Processing: {result_fd}')
		if 'fine_tuning' in result_fd:
			continue
	
		if idx == 0:
			# training data is set using first data
			trdata = pickle.load(open(f'{result_fd}/datasets/training_dataset.pickle', 'rb'))
			trsmiles = trdata.org_smi

		lstm_conditions = get_lstm_parameters(result_fd)	
		
		# model evaluation
		evaluated_metrics = evaluate_lstm_model(result_fd, trsmiles)
		lstm_conditions.update(evaluated_metrics)
		lstm_parameters[idx] = lstm_conditions 
	
		pd.DataFrame.from_dict(lstm_parameters, orient='index').to_csv(f'{outfd}/collected_lstm_metrics.tsv', sep='\t')
	# validtiy, uniqueness, novelty, properties (MW, logP, TPSA, nHBD, nHBA, Rings)

def evaluate_leadlike_model(bf, addprops=True):
	outfd		= MakeFolder(f'{bf}/results/analyze_leadlike', allow_override=True)
	lrs			= [0.0001, 0.0005]
	batches		= [64, 128]
	ret_all 	= dict()
	for lr, batch in product(lrs, batches): 
		result_fd 	= f'/home/miyao/work/projects/generative_models/results/lstm_trained_leadlike/lr{lr}_batch{batch}'
		trdata 		= pickle.load(open(f'{result_fd}/datasets/training_dataset.pickle', 'rb'))
		trsmiles 	= trdata.org_smi
		lstm_conditions = get_lstm_parameters(result_fd)	
		
		# model evaluation
		evaluated_metrics, smiles = evaluate_lstm_model(result_fd, trsmiles, return_smiles=True, nsamples=10000, add_properties=True)
		ret_all[f'{lr}_{batch}'] = evaluated_metrics
		pd.DataFrame.from_dict(ret_all, orient='index').to_csv(f'{outfd}/stats.tsv', sep='\t')

		# smiles evaluation
		outfname = f'{outfd}/lr{lr}_batch{batch}_sampled_smis.tsv'
		smiles.to_csv(outfname, sep='\t')
		WriteDataFrameSmilesToXls(smiles.sample(1000, random_state=0), 'canonical_smiles', outfname.replace('.tsv', '.xlsx'))

def evaluate_bimodal_model(bf, debug=False):
	outfd		= MakeFolder(f'{bf}/results/analyze_bimodals', allow_override=True)
	ret_all 	= dict()
	descs 		= ['RingCount','qed','MolWt','TPSA']
	weights		= ['equalweight_']
	nsamples 	= 10000
	for weight in weights:
		for desc in descs:
			result_fd 	= f'{results_fd}/smiles_multimodal/{weight}{desc}_lstm'
			trdata 		= pickle.load(open(f'{result_fd}/datasets/training_dataset.pickle', 'rb'))
			trsmiles 	= trdata.org_smi
			lstm_conditions = get_lstm_parameters(result_fd)	
			
			# model evaluation
			evaluated_metrics, smiles = evaluate_lstm_model(result_fd, trsmiles, return_smiles=True, nsamples=10000, add_properties=False)
			evaluated_metrics['ntraining'] = len(trdata)
			ret_all[f'{weight}{desc}'] = evaluated_metrics
			pd.DataFrame.from_dict(ret_all, orient='index').to_csv(f'{outfd}/stats.tsv', sep='\t')

			# smiles evaluation
			smicol = 'canonical_smiles'
			smiles = smiles.dropna(subset=smicol)
			descdf = calcRDKitDescriptors_specified(smiles[smicol], descs)
			smiles = smiles.join(descdf)
			if debug:
				smiles = smiles.sample(1000)
				trsmiles = trsmiles.sample(1000)
			try:
				genbits 	= smiles[smicol].apply(MorganbitCalcAsVectorFromSmiles)
			except:
				print('Error detected skip the descriptor:', desc)
				continue
			trbits 		= trsmiles.apply(MorganbitCalcAsVectorFromSmiles)
			genbits_np 	= np.vstack(genbits.values)
			trbits_np 	= np.vstack(trbits.values)
			nsplits 	= 40
			dfgenbits 	= pd.DataFrame(genbits_np, index=smiles.index)
			splitted 	= np.array_split(dfgenbits, nsplits)
			dists 		= Parallel(n_jobs=nsplits)([delayed(calc_min_dist_nnindices)(gen_batch, trbits_np) for gen_batch in splitted])
			df_dist 	= pd.concat(dists, axis=0)			
			df_dist.index = smiles.index # numpy
			smiles 		= smiles.join(df_dist)

			smiles['nn_tr_smiles'] = smiles['idxmins'].apply(lambda x: trsmiles.iloc[x])
			training_desc = calcRDKitDescriptors_specified(trdata.org_smi, [desc])
			makeHistogramTwo(smiles[desc], training_desc[desc], legend1=f'Generated: {len(smiles)}', 
							legend2=f'Training: {len(training_desc)}', 
							xlab_name=desc, save_fig_name=f'{outfd}/{weight}{desc}_histogram.png', is_scale=True,
							n_bins=100, y_title='')	
			outfname = f'{outfd}/{weight}{desc}_sampled_smis{nsamples}.tsv'
			smiles.to_csv(outfname, sep='\t')
			excel_sample = min(1000, len(smiles))
			WriteDataFrameSmilesToXls(smiles.sample(excel_sample, random_state=0), 
							 [smicol, 'nn_tr_smiles'], outfname.replace(f'{nsamples}.tsv', f'{excel_sample}.xlsx'))

def evaluate_bimodal_finetune_model(bf, nones=False, debug=False):
	suffix = '_nones' if nones else ''
	outfd		= MakeFolder(f'{bf}/results/analyze_bimodal_finetune{suffix}', allow_override=True)
	
	ret_all 	= dict()
	descs 		= ['RingCount','qed','MolWt','TPSA']
	weights		= ['equalweight_', '']
	nsamples 	= 10000
	
	for weight in weights:
		if nones and (weight ==''):
			continue
		for desc in descs:
			if nones:
				result_fd 	= f'{results_fd}/smiles_multimodal/fine_tuning_100epochs_lr00005/{weight}{desc}_lstm'
			else:
				result_fd 	= f'{results_fd}/smiles_multimodal/fine_tuning/{weight}{desc}_lstm'
			trdata 		= pickle.load(open(f'{result_fd}/datasets/training_dataset.pickle', 'rb'))
			trsmiles 	= trdata.org_smi
			
			# model evaluation
			evaluated_metrics, smiles = evaluate_lstm_model(result_fd, trsmiles, return_smiles=True, nsamples=10000, add_properties=False, use_last_epoch_model=nones)
			evaluated_metrics['ntraining'] = len(trdata)
			ret_all[f'{weight}{desc}'] = evaluated_metrics
			pd.DataFrame.from_dict(ret_all, orient='index').to_csv(f'{outfd}/stats.tsv', sep='\t')

			# smiles evaluation
			smicol = 'canonical_smiles'
			smiles = smiles.dropna(subset=smicol)
			descdf = calcRDKitDescriptors_specified(smiles[smicol], descs)
			smiles = smiles.join(descdf)
			training_desc = calcRDKitDescriptors_specified(trdata.org_smi, [desc])
			makeHistogramTwo(smiles[desc], training_desc[desc], legend1=f'Generated: {len(smiles)}', 
							legend2=f'Training: {len(training_desc)}', 
							xlab_name=desc, save_fig_name=f'{outfd}/{weight}{desc}_histogram.png', is_scale=True,
							n_bins=100, y_title='')	
			# similar training data identification 
			if debug:
				trsmiles  	= trsmiles.sample(1000)
				smiles		= smiles.sample(1000) # debug
			
			genbits 	= smiles[smicol].apply(MorganbitCalcAsVectorFromSmiles)
			trbits 		= trsmiles.apply(MorganbitCalcAsVectorFromSmiles)
			genbits_np 	= np.vstack(genbits.values)
			trbits_np 	= np.vstack(trbits.values)
			nsplits 	= 40
			dfgenbits 	= pd.DataFrame(genbits_np, index=smiles.index)
			splitted 	= np.array_split(dfgenbits, nsplits)
			dists 		= Parallel(n_jobs=nsplits)([delayed(calc_min_dist_nnindices)(gen_batch, trbits_np) for gen_batch in splitted])
			df_dist 	= pd.concat(dists, axis=0)			
			df_dist.index = smiles.index # numpy
			smiles 		= smiles.join(df_dist)
			smiles['nn_tr_smiles'] = smiles['idxmins'].apply(lambda x: trsmiles.iloc[x])
			outfname = f'{outfd}/{weight}{desc}_sampled_smis{nsamples}.tsv'
			smiles.to_csv(outfname, sep='\t')
			excel_sample = min(1000, len(smiles))
			# nn compounds sampling  + similarity 
			WriteDataFrameSmilesToXls(smiles.sample(excel_sample, random_state=0), ['canonical_smiles','nn_tr_smiles'], outfname.replace(f'{nsamples}.tsv', f'{excel_sample}.xlsx'))


def calc_min_dist_nnindices(dfx, y):
	dmat  		= pairwise_distances(dfx.values, y, metric='jaccard', n_jobs=1)
	idxmins  	= dmat.argmin(axis=1)
	nndistance 	= dmat.min(axis=1)
	retdf 		= pd.DataFrame.from_dict(dict(idxmins=idxmins, nn_distance=nndistance), orient='columns')
	retdf.index = dfx.index
	return retdf

def evaluate_bimodal_finetune_model_lr0005(bf):
	outfd		= MakeFolder(f'{bf}/results/analyze_bimodal_finetune_nones_lr0005', allow_override=True)
	
	ret_all 	= dict()
	descs 		= ['RingCount','qed','MolWt','TPSA']
	weights		= ['equalweight_']
	nsamples 	= 10000
	
	for weight in weights:
		for desc in descs:
			result_fd 	= f'{results_fd}/smiles_multimodal/fine_tuning_100epochs_lr00005/{weight}{desc}_lstm_lr0.005_nepochs100'
			trdata 		= pickle.load(open(f'{result_fd}/datasets/training_dataset.pickle', 'rb'))
			trsmiles 	= trdata.org_smi
			
			# model evaluation
			evaluated_metrics, smiles = evaluate_lstm_model(result_fd, trsmiles, return_smiles=True, nsamples=10000, add_properties=False, use_last_epoch_model=True)
			evaluated_metrics['ntraining'] = len(trdata)
			ret_all[f'{weight}{desc}'] = evaluated_metrics
			pd.DataFrame.from_dict(ret_all, orient='index').to_csv(f'{outfd}/stats.tsv', sep='\t')

			# smiles evaluation
			smiles = smiles.dropna(subset='canonical_smiles')
			if len(smiles) == 0:
				continue
			descdf = calcRDKitDescriptors_specified(smiles['canonical_smiles'], descs)
			smiles = smiles.join(descdf)
			training_desc = calcRDKitDescriptors_specified(trdata.org_smi, [desc])
			makeHistogramTwo(smiles[desc], training_desc[desc], legend1=f'Generated: {len(smiles)}', 
							legend2=f'Training: {len(training_desc)}', 
							xlab_name=desc, save_fig_name=f'{outfd}/{weight}{desc}_histogram.png', is_scale=True,
							n_bins=100, y_title='')	

			outfname = f'{outfd}/{weight}{desc}_sampled_smis{nsamples}.tsv'
			smiles.to_csv(outfname, sep='\t')
			excel_sample = min(1000, len(smiles))
			WriteDataFrameSmilesToXls(smiles.sample(excel_sample, random_state=0), 'canonical_smiles', outfname.replace(f'{nsamples}.tsv', f'{excel_sample}.xlsx'))



if __name__ == '__main__':
	if 0:
		evaluate_lstm_parameters(bf)
	if 0:
		evaluate_leadlike_model(bf)
	if 1:
		evaluate_bimodal_model(bf)
	if 0:
		evaluate_bimodal_finetune_model(bf)
	if 0:
		evaluate_bimodal_finetune_model(bf, nones=True) # none early stopping version.
	if 0:
		evaluate_bimodal_finetune_model_lr0005(bf) # last trial to change the number of epochs
