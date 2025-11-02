import logging
import torch
import torch.nn.functional as F
from .lstm import VanilaLSTM
from .tokenizer import SmilesTokenizer # might be necessary? 

class LSTMsampler():
	def __init__(self, model: VanilaLSTM, vocab, device, max_length=100, rseed=None):
		self.model 		= model.to(device)
		self.vocab 		= vocab
		self.device		= device
		self.max_length = max_length
		self.rseed 		= rseed
		self.init_rngs(self.rseed)
		# special characters	
		self.id_srt, self.id_end, self.id_pad = vocab.token2id[vocab.srt], vocab.token2id[vocab.end], vocab.token2id[vocab.pad]
	
	def greedy_sampling(self, n):
		# select the token with the highest probability
		pass

	def init_rngs(self, seed):
		if seed is None:
			return
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


	def probability_sampling(self, n):
		# based on the probabilities of tokens (multinominal distribution)
		self.model.eval()
		sampled_tokens = torch.zeros(n, self.max_length, dtype=torch.long, device=self.device)
		
		with torch.no_grad():
			tokenids = torch.tensor([self.id_srt]*n, dtype=torch.long, device=self.device).unsqueeze(1)
			states   = None
			sampled_tokens[:,0] = tokenids.squeeze()
			
			for i in range(1, self.max_length):
				out, states = self.model(tokenids, states, return_state=True)
				probs = F.softmax(out.squeeze(), dim=1) # (n, ntokens)
				next_tokenids = torch.multinomial(probs, 1)
				
				# save the tokens sampled
				sampled_tokens[:,i] = next_tokenids.squeeze().detach().clone()

				# if all batch reached the end of the sentence
				is_special_tokens = (next_tokenids == self.id_srt) | \
									(next_tokenids == self.id_end) | \
									(next_tokenids == self.id_pad)
				
				if is_special_tokens.sum() == len(next_tokenids):
					break
				
				tokenids = next_tokenids

			# convert to characters
			ltokens 	 = sampled_tokens.cpu().detach().tolist()
			sampled_smis = [SmilesTokenizer.untokenize_smiles(self.vocab.decode(sentence)) for sentence in ltokens]
		return sampled_smis


	def beam_sampling(self, n):
		raise NotImplementedError()
		


