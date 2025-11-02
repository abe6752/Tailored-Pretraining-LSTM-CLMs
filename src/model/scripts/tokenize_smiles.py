import pandas as pd
import sys
import pickle
from rdkit import Chem
from path_config import bf
from util.utility import MakeFolder, LogFile
from generative_models.tokenizer import create_vocabulary, count_freqency_tokens, is_eligible_smiles, count_moltokens
from generative_models.utils import Smiles2RDKitCanonicalSmiles
from chemistry.descriptors import HeavyAtomCount, calcRDKitDescriptors_specified, rdkit_descriptorlist

def sc1_tokenize_smiles(chembl_file, outfd: str, debug: bool=True):
	mols 		= pd.read_csv(chembl_file, sep='\t', index_col=0)
	if debug:
		mols = mols.sample(1000, random_state=0)
	
	outfd 		= MakeFolder(outfd, allow_override=True)
	logfile 	= LogFile(f'{outfd}/tokenize_log.txt')
	logfile.write(f'loaded molecules: {len(mols)}')
	smi_col = 'washed_openeye_smiles'

	# curateion by RDKit
	rdsmi_col = 'rdkit_smiles'
	mols[rdsmi_col] = mols[smi_col].apply(Smiles2RDKitCanonicalSmiles)
	ngmols = mols[rdsmi_col].isna().sum()
	logfile.write(f'cannnot handeld mols in rdkit: {ngmols}')
	okmols = mols[~mols[rdsmi_col].isna()]
	logfile.write(f'passed mols: {len(okmols)}')

	# molecular size filter # 99 percentile in the number of heavy atoms 
	okmols['nhatoms'] = okmols[rdsmi_col].apply(HeavyAtomCount)
	maxha, minha = okmols['nhatoms'].max(), okmols['nhatoms'].min()
	logfile.write(f'maximum heavy atom number is {maxha}')
	logfile.write(f'minimum heavy atom number is {minha}')

	# eliminate the 95 percentile in nha
	percentile = 0.95
	thres_nha  = okmols['nhatoms'].quantile(percentile)
	okmols 	   = okmols[okmols['nhatoms']<thres_nha]
	logfile.write(f'the nha percentile {percentile}')
	logfile.write(f'the nha threshold {thres_nha}')
	logfile.write(f'selected molecules: {len(okmols)}')

	# structural aleart is elimiated
	#filternames = ['Dundee', 'Glaxo', 'PAINS']
	filternames = ['Glaxo', 'PAINS']
	passmols = okmols[~okmols[filternames].any(axis=1)]
	failmols = okmols[okmols[filternames].any(axis=1)]
	logfile.write(f'passd {filternames}: {len(passmols)}')
	
	# tokenize molecules
	freq=count_freqency_tokens(passmols[rdsmi_col])
	pd.Series(freq, name='frequency').to_csv(f'{outfd}/token_frequency.tsv', sep='\t')
	outputcol= ['chembl_id', rdsmi_col] + filternames 
	passmols[outputcol].to_csv(f'{outfd}/passed_filters_rdkit_canonical_smiles.tsv', sep='\t')
	failmols[outputcol].to_csv(f'{outfd}/failed_filters_rdkit_canonical_smiles.tsv', sep='\t')

def sc2_select_eligible_mol_tokens(input_mols, token_file, outfd, debug=True):
	tokens = pd.read_csv(token_file, sep='\t', index_col=0)
	mols   = pd.read_csv(input_mols, sep='\t', index_col=0)
	mols   = mols.sample(1000) if debug else mols
	outfd  = MakeFolder(outfd, allow_override=True)
	logfp  = LogFile(f'{outfd}/token_selection_log.txt')
	
	# token threshold 
	fthres = 50
	logfp.write(f'curating tokens: frequency threshold: {fthres}')
	oktokens = tokens[tokens['frequency']>fthres]
	logfp.write(f'ok tokens: {len(oktokens)}')

	# select the eligible SMILES
	vocab = create_vocabulary(oktokens.index)
	mols['is_eligible'] = mols['rdkit_smiles'].apply(lambda x: is_eligible_smiles(x, vocab))
	okmols = mols[mols['is_eligible']]
	okmols['ntokens_srt_end'] = okmols['rdkit_smiles'].apply(lambda x: count_moltokens(x, include_begin_and_end=True))
	logfp.write(f'eligible smiles: {len(okmols)}')

	outputcol = ['chembl_id', 'rdkit_smiles', 'ntokens_srt_end']
	pickle.dump(vocab, open(f'{outfd}/vocab.pickle', 'wb'), protocol=pickle.DEFAULT_PROTOCOL)
	okmols[outputcol].to_csv(f'{outfd}/passed_mols.tsv', sep='\t')

def sc3_make_leadlike_data(inputfile, outfd, debug=True):
	outfd  = MakeFolder(outfd, allow_override=True)
	logfp  = LogFile(f'{outfd}/leadlike_selection_log.txt')
	
	mols = pd.read_csv(inputfile, sep='\t', index_col=0)
	if debug:
		mols = mols.sample(10000, random_state=0)
	
	logfp.write(f'loaded mols: {len(mols)}')
	# molecular weight calcualtion 
	descs = ['MolWt', 'MolLogP']
	ret   = calcRDKitDescriptors_specified(mols['rdkit_smiles'], descs, nworkers=8)
	
	# threshold 
	mw_low, mw_high = 100, 350
	logp_low, logp_high = 1.0, 3.0
	ret_ok = ret.query(f' {mw_low} <= MolWt <= {mw_high} and {logp_low} <= MolLogP <= {logp_high}')
	logfp.write(f'MW threshold (include end): ({mw_low}, {mw_high})')
	logfp.write(f'logp threshold (include end): ({logp_low}, {logp_high})')
	logfp.write(f'passed mols: {len(ret_ok)}')
	ok_mols = mols.join(ret_ok, how='inner')

	ok_mols.to_csv(f'{outfd}/leadlike_okmols{len(ok_mols)}.tsv', sep='\t')



if __name__ == '__main__':
	if 0:
		chembl_file = '/Users/miyao/work/datasets/chembl31/chemblx31-all/all_curated_cpds_chembl31.tsv'
		outfd = f'{bf}/results/vocabulary'
		sc1_tokenize_smiles(chembl_file, outfd, debug=False) 
	if 0:
		input_file = f'{bf}/results/vocabulary/passed_filters_rdkit_canonical_smiles.tsv'
		token_file = f'{bf}/results/vocabulary/token_frequency.tsv'
		outfd = f'{bf}/results/prepared_smiles'
		sc2_select_eligible_mol_tokens(input_file, token_file, outfd, debug=False)
	if 1:
		input_file = f'{bf}/results/prepared_smiles/passed_mols.tsv'
		outfd 	   = f'{bf}/results/leadlike_smiles'
		sc3_make_leadlike_data(input_file, outfd, debug=False)
