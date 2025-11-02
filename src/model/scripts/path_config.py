"""
Path configuration to the library
"""
import sys 
from util.utility import IsMac
if IsMac():
	bf 	= '/Users/miyao/work/research/generative_models'
	results_fd = f'{bf}/from_cluster/results'
	results_fd = f'{bf}/results'
else:
	bf 	= '/home/miyao/work/projects/generative_models'
	results_fd = f'{bf}/results'

sys.path.append(f'{bf}/generative_models') # base folder import 

