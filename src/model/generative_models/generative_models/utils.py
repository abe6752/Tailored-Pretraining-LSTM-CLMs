
from typing import Optional
from rdkit import Chem
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def importstr(module_str: str, from_: Optional[str]=None):
	"""
	>>> importstr('os')
	<module 'os' from '.../os.pyc'>
	"""
	if from_ is None and ':' in module_str:
		module_str, from_ = module_str.rsplit(':')
	module = __import__(module_str)
	for sub_str in module_str.split('.')[1:]: # loop over the rest of the modules
		module = getattr(module, sub_str)
	if from_:
		try: 
			return getattr(module, from_)
		except:
			raise ImportError(f'{module_str}.{from_}')
	return module

def run(app, *args):
	# like a commandline application
	argv = list(args)
	argv.insert(0, '--num_workers=4')
	log.info(f'Running: {app}({args}).main()')
	app_cls = importstr(*app.rsplit('.',1))
	app_cls(argv).main()
	log.info(f'Finished: {app}({args}).main()')

def Smiles2RDKitCanonicalSmiles(smi, **argv):
    """
    Standardize smiles to canonical smiles. If inappropriate, None is returned.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        try:
            return Chem.MolToSmiles(mol, **argv)
        except:
            log.info(f'Error in parsing {smi}...')
            return None
    else:
        return None

# added by Abe
def canonicalize_smiles(smiles_iter):
	cano_list = []
	seen = set()
	for smi in smiles_iter:
		if not isinstance(smi, str) or not smi:
			continue
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			continue
		cano = Chem.MolToSmiles(mol)
		if cano not in seen:
			seen.add(cano)
			cano_list.append(cano)
	return cano_list