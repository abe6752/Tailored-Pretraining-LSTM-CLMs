import logging
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class VanilaGPT(nn.Module):
	def __init__(self, ntokens,
			  nblocks=12,
			  nheads=12,
			  nembed=768, # this is amplieifed by nheads inside the GPT
			  pe_dropout=0.1, # positional encoder
			  at_dropout=0.2, # attention block dropout
			  padding_idx=None
			  ):
		super().__init__()
		# save model architecture
		self.params = namedtuple('Params', ['ntokens, nblocks, nheads, nembed, pe_dropout, at_droupout, padding_idx'])
		self.params.ntokens = ntokens
		self.params.nblocks = nblocks
		self.params.nheads  = nheads # multi heads
		self.params.nembed  = nembed # nembed // nheads = dimention per head 
		self.params.pe_dropout = pe_dropout
		self.params.at_dropout = at_dropout
		self.params.padidx  = padding_idx
		
		# First embedding and positional encoder
		self.tok_embed = nn.Embedding(ntokens, nembed, padding_idx=padding_idx)
		self.pos_embed = PositionalEncoding(nembed, pe_dropout)

		# Block layers (self attention + MLP)
		self.blocks = nn.ModuleList([DecoderBlock(nembed, nheads, at_dropout, nembed) for _ in range(nblocks)])

		# Final layer
		self.norm 	= nn.LayerNorm(nembed)
		self.linear	= nn.Linear(nembed, ntokens)

		self.apply(self.init_weights_)
	
	@torch.no_grad()
	def init_weights_(self, module):
		"""
		Copied from the code to initialize weight: 
		https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
		"""
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			torch.nn.init.zeros_(module.bias)
			torch.nn.init.ones_(module.weight)

	
	def forward(self, input):
		out = self.tok_embed(input)
		out = self.pos_embed(out)
		tgt_mask = make_masked_input(out.size(-2)).to(self.device) #batch first (Batch, Sentence, Word)
		pad_mask = make_padding_mask(input, self.padidx).to(self.device)
		for block in self.blocks:
			out = block(out, tgt_mask, pad_mask)
		out = self.norm(out)
		out = self.linear(out)
		return out

class PositionalEncoding(nn.Module):
	# batch first data is assumed (Batch, Sequence, embedding)
	def __init__(self, inputdim, dropout=0.1, maxlen=1000):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		pe  = torch.zeros(maxlen, inputdim)
		pos = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
		coef = torch.exp(torch.arange(0, inputdim, 2).float()*(-np.log(1e4)/inputdim)) 
		pe[:, 0::2] = torch.sin(pos * coef)
		pe[:, 1::2] = torch.cos(pos * coef)
		pe.unsqueeze_(0) # add batch 
		self.register_buffer('pe', pe) # like a constant value
	
	def forward(self, x):
		x = x + self.pe[:,:x.size(-2),:] # pe is the batch shape
		x.squeeze_() # for non batch 
		return self.dropout(x)


class DecoderBlock(nn.Module):
	def __init__(self, inputdim, nheads, dropout, mlp_hdim):
		super().__init__()

		self.self_att	= nn.MultiheadAttention(inputdim, nheads, dropout, batch_first=True) #self attention
		self.norm1 	   	= nn.LayerNorm(inputdim)
		self.dropout1	= nn.Dropout(dropout)
		self.linear1	= nn.Linear(inputdim, mlp_hdim)
		self.act 		= nn.GELU()
		self.linear2	= nn.Linear(mlp_hdim, inputdim)
		self.norm2		= nn.LayerNorm(inputdim)
		self.dropout2	= nn.Dropout(dropout)
	
	def forward(self, x, target_mask, padding_mask):
		att_out, att_map = self.self_att(x, x, x, key_padding_mask=padding_mask, attn_mask=target_mask)
		x = x + self.dropout1(att_out)
		x = self.norm1(x)
		ff_out = self.linear2(self.act(self.linear1(x))) # MLP
		x = x + self.dropout2(ff_out)
		x = self.norm2(x)
		return x

### must be chosen to float one!! Torch bug (or jsut warning)
def make_masked_input(x_size):
	# this cannot be incorporated in the VanilaGPT.__init__()
	mask = torch.triu(torch.ones(x_size, x_size), diagonal=1)
	mask = mask.float().masked_fill(mask==1, float('-inf')).masked_fill(mask==0, float('0.0'))
	return mask
	
def make_padding_mask(input, paddingidx):
	if paddingidx is None:
		return None 
	mask = input.detach() == paddingidx
	mask = mask.float().masked_fill(mask==1, float('-inf')).masked_fill(mask==0, float('0.0'))
	return mask

def save_data(self):
	return self.params._asdict()