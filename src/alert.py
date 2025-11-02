# getting structural alerts 
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from joblib import Parallel, delayed, cpu_count

def HasStructuralAlert(mol: Chem.Mol):
	cname = 'ALL'
	catalog = FilterCatalogParams.FilterCatalogs.names[cname]
	params = FilterCatalogParams()
	params.AddCatalog(catalog)
	catalog = FilterCatalog(params)
	return catalog.HasMatch(mol)

def GetStructuralAlerts(pd_romols: pd.Series):
	catalog_names = list(FilterCatalogParams.FilterCatalogs.names.keys())
	catalog_names.remove('ALL')
	ret_pdmols = pd.DataFrame(index=pd_romols.index)
	for cname in catalog_names:
		catalog = FilterCatalogParams.FilterCatalogs.names[cname]
		params = FilterCatalogParams()
		params.AddCatalog(catalog)
		catalog = FilterCatalog(params)
		ret_pdmols[cname] = pd_romols.apply(lambda x: catalog.HasMatch(x))
	return ret_pdmols


def GetStructuralAlertsMT(pd_romols: pd.Series, njobs=None):
	if (njobs == None) or (njobs == -1):
		njobs = cpu_count() -1 
	splitted = np.array_split(pd_romols, njobs)
	res = Parallel(n_jobs=njobs)([delayed(CalcAlertWorker)(mols, wid) for wid, mols in enumerate(splitted)])
	cat_desc = pd.concat(res) # order might not be different
	return cat_desc.loc[pd_romols.index] 

def GetDescription_MatchedAtoms(mol):
	catalog_names = list(FilterCatalogParams.FilterCatalogs.names.keys())
	catalog_names.remove('ALL')
	ret_inf = dict()
	for cname in catalog_names:
		catalog = FilterCatalogParams.FilterCatalogs.names[cname]
		params = FilterCatalogParams()
		params.AddCatalog(catalog)
		catalog = FilterCatalog(params)
		catalog_list = catalog.GetMatches(mol)
		catalogs = dict()
		if len(catalog_list) == 0:
			continue
		for idx, catalog in enumerate(catalog_list):
			hitinf = dict()
			hitinf['description'] 	= catalog.GetProp('description')
			hitinf['Scope']			= catalog.GetProp('Scope')
			filter = catalog.GetFilterMatches(mol)
			hitinf['atomidx']		= [x[1] for x in filter[0].atomPairs]
			hitinf['smarts']		= Chem.MolToSmarts(filter[0].filterMatch.GetPattern())
			catalogs[idx] = hitinf
		ret_inf[cname]=catalogs
	return ret_inf


def CalcAlertWorker(pd_mols, wid):
	catalog_names = list(FilterCatalogParams.FilterCatalogs.names.keys())
	catalog_names.remove('ALL')
	ret_pdmols = pd.DataFrame(index=pd_mols.index)
	for cname in catalog_names:
		if wid == 0:
			print(f'processing: {cname}', end='\r')
		catalog = FilterCatalogParams.FilterCatalogs.names[cname]
		params = FilterCatalogParams()
		params.AddCatalog(catalog)
		catalog = FilterCatalog(params)
		ret_pdmols[cname] = pd_mols.apply(lambda x: catalog.HasMatch(x))
	return ret_pdmols




	

