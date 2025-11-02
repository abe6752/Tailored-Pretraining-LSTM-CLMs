import re
import numpy as np
import pandas as pd
import logging

from collections import Counter, OrderedDict
from typing import List
from dataclasses import dataclass
from rdkit import Chem

log = logging.getLogger(__name__)



"""
The codes were modified from vocabulary.py from DeepAC under the license: Creative Commons Attribution 4.0 International
Original codes can be found https://doi.org/10.5281/zenodo.7153115 the authors: Hengwei Chen, Martin Vogt,Jürgen Bajorath.
The tokens and index are separately crated not in a single dictionary
"""
@dataclass(frozen=True)
class special_tokens:   
    pad     = 'PAD'
    start   = 'START'
    end     = 'END'

class Vocabulary:
    """
    Stores the tokens and their conversion to one-hot vectors.
    Modified to store token information in two independent dictionaries. 
    delete option was eliminated because it would make unordered indices (or chaning the indices)
    """
    pad = special_tokens.pad
    srt = special_tokens.start
    end = special_tokens.end

    def __init__(self, tokens: List=None, starting_id: int=0):
        self.token2id = OrderedDict()
        self.id2token = OrderedDict()
        self.lastidx  = starting_id-1 # add 1 inside 
        # set from currently tested 
        if tokens:
            for token in tokens:
                self.add(token) # if true add the token successfully
            
    def __getitem__(self, token): # only to get ids 
        return self.token2id[token]
    
    def get_token(self, id):
        return self.id2token[id]

    def add(self, token):
        """
        Adds a token if possible, return True else False
        """
        if not isinstance(token, str):
            print(f"Token must be a string, but the type is {type(token)}")
            print('Do nothing.')
            return False
        if token in self.token2id:
            print("Token already present in the vocabulary")
            print('Do nothing.')
            return False
        self._add(token)
        return True

    def add_tokens(self, tokens):
        return [self.add(token) for token in tokens]

    def __contains__(self, token_or_id):
        return token_or_id in self.token2id

    def __eq__(self, other_vocabulary):
        return self.token2id == other_vocabulary.token2id

    def __len__(self):
        return len(self.token2id)

    def encode(self, tokens):
        """
        Encodes a list of tokens to a 1-hot encoded vectors.
        """
        ohe_vect = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            ohe_vect[i] = self.token2id[token]
        return ohe_vect

    def decode(self, ohe_vect):
        """
        Decodes a one-hot encoded vector matrix to a list of tokens.
        """
        tokens = []
        for idx, ohv in enumerate(ohe_vect):
            token = self.id2token[ohv]
            if (token == self.end) or (token == self.pad):
                break
            if token == self.srt:
                if idx != 0:
                    break
                else:
                    continue
            tokens.append(self.id2token[ohv])
        
        return tokens

    def _add(self, token):
        if token not in self.token2id:
            self.lastidx +=1
            self.token2id[token] = self.lastidx
            self.id2token[self.lastidx] = token
        else:
            print("token is already in the vocabulary. Do nothing")

class SmilesTokenizer:
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
    pad = special_tokens.pad
    srt = special_tokens.start
    end = special_tokens.end

    @staticmethod
    def tokenize_smiles(data, with_begin_and_end=True):
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = SmilesTokenizer.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens
        tokens = split_by(data, SmilesTokenizer.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = [special_tokens.start] + tokens + [special_tokens.end]
        return tokens
    
    @staticmethod
    def untokenize_smiles(tokens):
        smi = ""
        for token in tokens:
            if token in [special_tokens.pad, special_tokens.end]:
                break
            if token != special_tokens.start:
                smi += token
        return smi

"""
API functions
"""
def count_freqency_tokens(pd_smiles: pd.Series):
    tokens = Counter()
    for smi in pd_smiles:
        tokens.update(SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=False))
    return dict(sorted(tokens.items(), key=lambda x: x[1], reverse=True))

def create_vocabulary(tokens: list):
    """
    Creates a vocabulary for a list of tokens.
    """
    vocabulary = Vocabulary()
    tokens = [str(tk) for tk in tokens]
    vocabulary.add_tokens([special_tokens.pad, special_tokens.start, special_tokens.end] + sorted(tokens))  # pad=0, start=1, end=2
    return vocabulary

def is_eligible_smiles(smi, vocabulary):
    tokens = SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=False)
    return sum([token in vocabulary for token in tokens]) == len(tokens)

def count_moltokens(smi, include_begin_and_end=False):
    tokens = SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=include_begin_and_end)
    return len(tokens)
    
