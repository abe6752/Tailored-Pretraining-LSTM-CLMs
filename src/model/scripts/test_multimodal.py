import pandas as pd 
import numpy as np
import random
from copy import deepcopy
from path_config import results_fd, bf
from chemistry.descriptors import calcRDKitDescriptors_specified, rdkit_descriptorlist
from util.utility import MakeFolder, GetRootLogger
from scipy.stats import norm, gaussian_kde
from figure.plots import MakeLinePlotsSeaborn, makeHistogramOne

from generative_models.training import GenerativeModelTrainer
import matplotlib.pyplot as plt



def make_bimodal_trdata(debug=True):
	outfd = MakeFolder(f'{results_fd}/smiles_multimodal', allow_override=True)
	curated_smiles = pd.read_csv(f'{results_fd}/prepared_smiles/passed_mols.tsv', sep='\t', index_col=0)
	logger =GetRootLogger(f'{outfd}/loggging_inf.txt', clearlog=True)
	logger.info(f'Load mols: {len(curated_smiles)}')
	if debug:
		curated_smiles = curated_smiles.sample(n=10000)
	descs = ['RingCount','qed','MolWt','TPSA']
	dfdescs = calcRDKitDescriptors_specified(curated_smiles['rdkit_smiles'], descs, nworkers=40)
	curated_smiles = curated_smiles.join(dfdescs)
	curated_smiles.to_csv(f'{outfd}/smiles_descriptors.tsv', sep='\t')
	
def sampling_multimodal(debug=True):
	# test case 1
	outfd = MakeFolder(f'{results_fd}/smiles_multimodal/sampling', allow_override=True) 
	mols = pd.read_csv(f'{results_fd}/smiles_multimodal/smiles_descriptors.tsv', sep='\t', index_col=0)
	# for each descriptor
	descs = ['RingCount','qed','MolWt','TPSA']
	n = 100000 if not debug else 2000
	for desc in descs:
		dmax, dmin = mols[desc].max(), mols[desc].min()
		q1, q3 = mols[desc].quantile(0.10),  mols[desc].quantile(0.95)
		sigma  = mols[desc].std()
		gmix   = MakeMixtureGaussians([q1, q3], sigma*0.1) #misterios scaling 
		xline  = np.linspace(dmin, dmax, 1000)
		data   = pd.DataFrame(xline[:,np.newaxis], columns=[desc])
		data['pdf'] = data[desc].apply(gmix)
		if 0:
			MakeLinePlotsSeaborn(data, desc, 'pdf', save_fig_name=f'{outfd}/pdf_example.png')	
			makeHistogramOne(mols[desc], save_fig_name=f'{outfd}/{desc}_histogram.png', xlab_name=desc, show_legend=False)	

		# sampling MWs based on the pdf
		mw_samples = SampleOnPDF(gmix, mols[desc], n)
		if 0:
			makeHistogramOne(mw_samples, save_fig_name=f'{outfd}/{desc}_sampled_{len(mw_samples)}histogram.png', xlab_name=desc, show_legend=False)	
			mols.loc[mw_samples.index].to_csv(f'{outfd}/{desc}_sampled{len(mw_samples)}.tsv', sep='\t')

		# change the heights of peaks (assign cluster)
		mw_class = mw_samples.apply(lambda x: np.argmax(gmix(x, return_class=True)))
		# equal weight / opposite weight
		# equal weight
		zero_class = mols.loc[mw_class[mw_class==0].index]
		one_class  = mols.loc[mw_class[mw_class==1].index]
		nzero, none = len(zero_class), len(one_class)
		
		# sampling ratio is monitored (down sampling. to avoid augmentation ambiguity)
		if nzero > none: 
			zero_class_aug = zero_class.sample(none, replace=False, random_state=0)
			same_weight = pd.concat([zero_class_aug, one_class], ignore_index=True)	
		else:
			one_class_aug = one_class.sample(none, replace=False, random_state=0)
			same_weight = pd.concat([zero_class, one_class_aug], ignore_index=True)	
		
		makeHistogramOne(same_weight[desc], save_fig_name=f'{outfd}/sameweight_{desc}_sampled_{len(mw_samples)}histogram.png', xlab_name=desc, show_legend=False)	
		same_weight.to_csv(f'{outfd}/sameweight_{desc}.tsv', sep='\t')	
		


def SampleOnPDF(pdf, samples: pd.Series, n, use_kdf=False, small_thres=1e-5):
	if len(samples) < n:
		raise ValueError('sampling points are greater than the input data.')

	# random sampling
	if use_kdf:
		spdf 	= gaussian_kde(samples)
	else:
		spdf 	= lambda x: 1
	samples = samples.sample(frac=1, random_state=42) # distribution is considered
	count = 0 
	xprev = samples[0]
	sampled = []

	for idx,x in samples[1:].items():
		print(f'sampled {count}', end='\r')
		if pdf(x) < small_thres: # too small
			continue
		r = pdf(x)*spdf(xprev)/(pdf(xprev)*spdf(x))
		if r > 1:
			sampled.append(idx)
			xprev=x
			count+=1
		else:
			prob = r
			if random.uniform(0,1) < prob:
				sampled.append(idx)
				xprev=x
				count+=1
		if count >= n:
			break
	return samples.loc[sampled]


def MakeMixtureGaussians(centers, sigmas, weights=None):
	ngauss = len(centers)
	if not isinstance(sigmas, list): # if univariance is used
		sigmas = ngauss*[sigmas]
	if weights is None:
		weights = ngauss*[1/ngauss]
	def gaussian_mixture(x, return_class=False):
		cvals = [norm.pdf(x, centers[i], sigmas[i])*weights[i] for i in range(ngauss)]
		if return_class:
			return cvals
		else:
			return sum(cvals)
	return gaussian_mixture


def run_lstm_training(debug=True):
	smiles_fd 	= f'{results_fd}/smiles_multimodal/sampling'
	descs 		= ['RingCount','qed','MolWt','TPSA']
	smi_colname ='rdkit_smiles'
	nworker 	= 1
	vocab 		= f'{results_fd}/prepared_smiles/vocab.pickle'
	nepochs 	= 1 if debug else 100
	args = [
		'--model','lstm',
		'--epochs',str(nepochs), 
		'--num-workers',str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--validation-ratio','0.05',
		'--exclude-pad-loss','1',
		'--early-stopping-patience','10',
		'--smi-colname',smi_colname,
		'--vocab', vocab,
		'--nlayers', '3',
		'--hidden-dim', '256',
		'--embed-dim', '256',
		'--layernorm',
		'--lr', '0.0005',
		'--batch-size', '64',
		'--no-write-xlsx'
		]
	for desc in descs:
		if desc in ['RingCount', 'qed']:
			continue
		# equal weight 
		outfd 	= f'{results_fd}/smiles_multimodal/equalweight_{desc}_lstm'
		smiles 	= f'{smiles_fd}/sameweight_{desc}.tsv'
		curr_args = deepcopy(args)
		curr_args.extend(['--outdir', outfd])
		curr_args.extend(['--data',smiles])
		lstm_trainer = GenerativeModelTrainer(curr_args)
		lstm_trainer.main()

	for desc in descs:
		outfd 	= f'{results_fd}/smiles_multimodal/{desc}_lstm'
		smiles 	= f'{smiles_fd}/{desc}_sampled100000.tsv'
		curr_args = deepcopy(args)
		curr_args.extend(['--outdir', outfd])
		curr_args.extend(['--data',smiles])
		lstm_trainer = GenerativeModelTrainer(curr_args)
		lstm_trainer.main()


def run_lstm_training_finetune(debug=True):
	"""
	Fine tune is tested instead of training from scratch to compenste a limited number of training samples
	"""
	smiles_fd 	= f'{results_fd}/smiles_multimodal/sampling'
	descs 		= ['RingCount','qed','MolWt','TPSA']
	smi_colname ='rdkit_smiles'
	nworker 	= 20

	modelfd	= f'{bf}/results/vanila_lstm/lr0005_embed256_layer4_hdim512'
	mstr	= f'{modelfd}/model/best_model_structure_epoch16.pickle'
	mstate 	= f'{modelfd}/model/best_model_epoch16.pth'
	vocab	= f'{modelfd}/model/vocabulary.pickle'

	nepochs 	= 1 if debug else 40
	lr 			= 0.0001
	batch 		= 128
	args = [
		'--model','lstm',
		'--epochs',str(nepochs), 
		'--num-workers',str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--validation-ratio','0.0', 
		'--exclude-pad-loss','1',
		#'--early-stopping-patience','10',
		'--smi-colname',smi_colname,
		'--vocab', vocab,
		'--load-model',
		'--model-structure', mstr,
		'--model-state', mstate,
		'--lr', str(lr),
		'--batch-size', str(batch),
		'--no-write-xlsx'
		]
	baseoutfd = MakeFolder(f'{results_fd}/smiles_multimodal/fine_tuning_non_es', allow_override=True) 
	for desc in descs:
		# equal weight 
		outfd 	= f'{baseoutfd}/equalweight_{desc}_lstm'
		smiles 	= f'{smiles_fd}/sameweight_{desc}.tsv'
		curr_args = deepcopy(args)
		curr_args.extend(['--outdir', outfd])
		curr_args.extend(['--data',smiles])
		lstm_trainer = GenerativeModelTrainer(curr_args)
		lstm_trainer.main()
	if 0:
		for desc in descs:
			outfd 	= f'{baseoutfd}/{desc}_lstm'
			smiles 	= f'{smiles_fd}/{desc}_sampled100000.tsv'
			curr_args = deepcopy(args)
			curr_args.extend(['--outdir', outfd])
			curr_args.extend(['--data',smiles])
			lstm_trainer = GenerativeModelTrainer(curr_args)
			lstm_trainer.main()
def run_lstm_training_finetune_2nd(lr, nepochs):
	"""
	Fine tune is tested instead of training from scratch to compenste a limited number of training samples
	"""
	smiles_fd 	= f'{results_fd}/smiles_multimodal/sampling'
	descs 		= ['RingCount','qed','MolWt','TPSA']
	smi_colname ='rdkit_smiles'
	nworker 	= 20

	modelfd	= f'{bf}/results/vanila_lstm/lr0005_embed256_layer4_hdim512'
	mstr	= f'{modelfd}/model/best_model_structure_epoch16.pickle'
	mstate 	= f'{modelfd}/model/best_model_epoch16.pth'
	vocab	= f'{modelfd}/model/vocabulary.pickle'

	batch 		= 128
	args = [
		'--model','lstm',
		'--epochs',str(nepochs), 
		'--num-workers',str(nworker),
		'--override-folder','1', 
		'--sampling-epoch','1000',
		'--tensorboard-prefix','tf',
		'--validation-ratio','0.0', 
		'--exclude-pad-loss','1',
		#'--early-stopping-patience','10',
		'--smi-colname',smi_colname,
		'--vocab', vocab,
		'--load-model',
		'--model-structure', mstr,
		'--model-state', mstate,
		'--lr', str(lr),
		'--batch-size', str(batch),
		'--no-write-xlsx',
		'--save-snapshot-models','1,5,25,50,75'
		]
	
	baseoutfd = MakeFolder(f'{results_fd}/smiles_multimodal/fine_tuning_{nepochs}epochs_lr{lr}', allow_override=True) 
	for desc in descs:
		outfd 	= f'{baseoutfd}/equalweight_{desc}_lstm'
		smiles 	= f'{smiles_fd}/sameweight_{desc}.tsv'
		curr_args = deepcopy(args)
		curr_args.extend(['--outdir', outfd])
		curr_args.extend(['--data',smiles])
		lstm_trainer = GenerativeModelTrainer(curr_args)
		lstm_trainer.main()

if __name__ =='__main__':
	if 0:
		make_bimodal_trdata(debug=False)
	if 0:
		sampling_multimodal(debug=False)
	if 0:
		run_lstm_training(debug=False)
	if 0:
		run_lstm_training_finetune(debug=False)
	if 1:
		run_lstm_training_finetune_2nd(lr=0.0005, nepochs=100)