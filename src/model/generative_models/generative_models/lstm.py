from collections import namedtuple
import logging
import torch.nn as nn

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class VanilaLSTM(nn.Module):
	"""
	Simple LSTM model to generate probabilities of tokens
	"""		
	def __init__(self, ntokens, 
			  embed_size=256, 
			  hidden_size=512, 
			  nlayers=3, 
			  dropout_lstm=0.2,
			  layernorm=True,
			  padding_idx=None
			  ):
		super().__init__()	
		Params 		= namedtuple('Params', ['ntokens', 'embed_size', 'hidden_size', 'nlayers','dropout_lstm', 'layernorm', 'padding_idx'])
		self.params = Params(ntokens, embed_size, hidden_size, nlayers, dropout_lstm, layernorm, padding_idx)		
		self.embed 	= nn.Embedding(num_embeddings=ntokens, embedding_dim=embed_size, padding_idx=padding_idx) ### padding idx should be specified?
		self.lstm  	= nn.LSTM(input_size=embed_size,
					  		hidden_size=hidden_size, 
							num_layers=nlayers,
							dropout=dropout_lstm, 
							batch_first=True
							)
		self.linear = nn.Linear(hidden_size, ntokens)
		self.layernorm = layernorm
		if layernorm:
			self.norm  	= nn.LayerNorm(hidden_size, eps=0.001)
		self._init_weights()	
	
	def _init_weights(self):
		nn.init.xavier_uniform_(self.embed.weight)
		nn.init.xavier_uniform_(self.linear.weight)
		nn.init.uniform_(self.linear.bias)
		# set intial weight for lstm hidden layers
		for weight in self.lstm._all_weights:
			if 'weight' in weight:
				nn.init.xavier_uniform_(getattr(self.lstm,weight))
			elif 'bias' in weight:
				nn.init.uniform_(getattr(self.lstm, weight))
		
	def forward(self, input, input_state=None, return_state=False):
		self.lstm.flatten_parameters()
		out = self.embed(input)
		out, state = self.lstm(out, input_state)
		if self.layernorm:
			out = self.norm(out)
		out = self.linear(out)
		if return_state:
			return out, state
		else:
			return out
		
	def save_data(self):
		return self.params._asdict() # necessary to avoid pickling error

	


		
	