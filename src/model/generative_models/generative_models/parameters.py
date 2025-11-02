import argparse

def set_global_configs_(parser: argparse.ArgumentParser):
	parser.add_argument('--num-workers',
					help='Number of workers',
					default=1,
					type=int
					)
	parser.add_argument('--outdir',
					help='Folder where output results will be stored.',
					default='./results/lstm_model',
					)
	
	parser.add_argument('--case',
					help='subdirecotry where outdir represents a series of experiments',
					default='',
					type=str
					)
		
	parser.add_argument('--override-folder',
					help='Allow override the existing folder with the same name.',
					default=0,
					type=int
					)
	parser.add_argument('--tensorboard-prefix',
					help='Prefix to be used for Tensor board setting.',
					default='tensor_board',
					type=str
					)
	parser.add_argument('--data',
					help='tsv file containing SMILES to be tokenized and used for training.',
					type=str
					)
	parser.add_argument('--smi-colname',
					help='Column name of smiles in the tsv file.',
					type=str
					)
	parser.add_argument('--vocab',
					help='pickle file containing Vocaburary of tokens.',
					type=str
					)
	parser.add_argument('--random-seed',
					help="Random seed for train/validation split",
					default=42,
					type=int
					)
	parser.add_argument('--debug',
					help='Conduct debuggin or not',
					action=argparse.BooleanOptionalAction,
					)
	parser.add_argument('--sampling-epoch',
					help='do generation (sampling) at the end of each epoch',
					default=100,
					type=int
					)
	parser.add_argument('--model',
					help='Model name: either [lstm or gpt]',
					default='lstm',
					choices=['lstm','gpt']
					)
	parser.add_argument('--write-xlsx',
					 help='Suppress xlsx file outputs during sampling',
					 action=argparse.BooleanOptionalAction
					 )
	parser.add_argument('--save-snapshot-models',
					help='Save model at specific epochs during training. Comma separated strings are input. "10,20,30" ',
					type=str,
					default=''
					)

def set_training_conditions_(parser: argparse.ArgumentParser):
	parser.add_argument('--batch-size',
					help='Batch size for training',
					default=32,
					type=int
					)
	parser.add_argument('--epochs',
					help='Epochs',
					default=10,
					type=int
					)
	parser.add_argument('--validation-ratio',
					help='Training and validation data set ratio',
					default=0.1,
					type=float
					)
	parser.add_argument('--lr',
					help='Learning rate',
					default=0.001,
					type=float
					)
	parser.add_argument('--exclude-pad-loss',
					help='exclude padding loss',
					default=1,
					type=int
					)
	parser.add_argument('--early-stopping-patience',
					help='early stopping based on the validation loss',
					default=0,
					type=int
					)
	parser.add_argument('--load-model',
					 help='Load pre-trained model from a file',
					 action=argparse.BooleanOptionalAction
					 )
	parser.add_argument('--model-structure',
				help='Path to a model structure file.',
				type=str
				)
	parser.add_argument('--model-state',
				help='Path to a model state file (must be a pair of the "--model-structure" param)',
				type=str
				)
	parser.add_argument('--standardize-smiles',
				help='Standardize smiles and token checks are conducted. It might reduce the calculation speed.',
				action=argparse.BooleanOptionalAction
				)


def set_lstm_conditions_(parser: argparse.ArgumentParser):
	parser.add_argument('--embed-dim',
				help='Dimension of embedding',
				default=128,
				type=int
				)
	parser.add_argument('--dropout-ratio',
				help='dropout-ratio',
				default=0.2,
				type=float
				)
	parser.add_argument('--nlayers',
				help='Number of LSTM layers',
				default=3,
				type=int
				)
	parser.add_argument('--hidden-dim',
				help='Hidden layer size for LSTM',
				default=256,
				type=int
				)
	parser.add_argument('--layernorm',
				help='User layernormalization',
				action=argparse.BooleanOptionalAction
				)

def set_gpt_conditions_(parser: argparse.ArgumentParser):
	parser.add_argument('--embed-dim',
				help='Dimension of embedding',
				default=768,
				type=int
				)
	parser.add_argument('--pe-dropout-ratio',
				help='dropout-ratio',
				default=0.1,
				type=float
				)
	parser.add_argument('--nblocks',
				help='Number of Block layers in GPT',
				default=12,
				type=int
				)
	parser.add_argument('--nheads',
				help='Multi head attention in GPT',
				default=12,
				type=int
				)

def set_finetune_conditions_(parser: argparse.ArgumentParser):
	parser.add_argument('--save-epoch-models',
				help='Whether to save the model per epoch or not',
				action=argparse.BooleanOptionalAction
				)
	parser.add_argument('--use-cpus',
				help='Force to use CPUs instead of available GPUs.',
				action=argparse.BooleanOptionalAction,
				)
	