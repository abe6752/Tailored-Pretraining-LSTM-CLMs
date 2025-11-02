import os
import sys
import time
import logging
import datetime


def GetTime(return_second=False, return_date=False):
	t = datetime.datetime.fromtimestamp(time.time())
	if return_second:
		return '{}{}{}_{}{}{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
	elif return_date:
		return '{}{}{}'.format(t.year,t.month,t.day)
	else:
		return '{}{}{}_{}{}'.format(t.year, t.month, t.day, t.hour, t.minute)
	

def search_exist_suffix(f_path):
	dirname, basename = os.path.split(f_path)

	for i in range(1,1000):
		if '.' in basename: # file
			n_name = basename.replace('.', '_{}.'.format(i))
		else:
			n_name = basename + '_' + str(i) # folder
		
		new_name = os.path.join(dirname,n_name)
		if not os.path.exists(new_name):
			return new_name
		# could not find unused filename for 1000 loops
	raise ValueError('cannot find unused folder names')


def MakeFolder(folder_path, allow_override=False, skip_create=False, time_stamp=False):
	"""
	Make a folder
	"""
	today = GetTime(return_date=True)
	if os.path.exists(folder_path) and skip_create:
		if time_stamp:
			return f'{folder_path}_{today}'
		return folder_path

	if os.path.exists(folder_path) and (not allow_override):
		Warning('Specified folder already exists. Create new one')
		folder_path = search_exist_suffix(folder_path)
		
	if time_stamp:
		folder_path = f'{folder_path}_{today}'
		
	os.makedirs(folder_path, exist_ok=allow_override)

	return folder_path


def GetRootLogger(file_name, 
                showstd: bool=True, 
                simple_format: bool=True, 
                clearlog: bool=False,
                clearhandler: bool=True
                ):
    """
    Wrapper function for setting logger (memorize utility information)
    """
    logger = logging.getLogger() # get the root loger
    if clearhandler: 
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
        
    logger.setLevel(logging.INFO)
    mode = 'w' if clearlog else 'a'
    fh = logging.FileHandler(file_name, mode=mode)
    fh.setLevel(logging.INFO)
    if not simple_format:
        f ='[%(levelname)-8s] [pid:%(process)d] [%(name)s]:[%(lineno)03d]:[%(message)s]'
    else:
        f = '%(asctime)s\t%(name)s\t%(message)s'
    fmt = logging.Formatter(f)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if showstd:
        hd = logging.StreamHandler(sys.stdout)
        hd.setLevel(logging.INFO)
        logger.addHandler(hd)
        
    logger.info('Start logging')
    return logger