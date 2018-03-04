
# coding: utf-8

# In[ ]:




from __future__ import print_function, unicode_literals

import os
import sys
import codecs
import random
import pickle
from time import time
import numpy as np
import string
import re
import nltk
from nltk import Tree
from collections import defaultdict
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

from CYK_parser import CYK_parser


# In[ ]:


if len(sys.argv) == 2: # Interactive mode
    MODE = "shell"
    MODEL_DIR = sys.argv[1]
elif len(sys.argv) == 4: # Batch mode
    MODE = "f"
    MODEL_DIR = sys.argv[1]
    INPUT_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]
else:
    print ("Parameter error!\n")
    print ("Please specify the path to PCFG model, or occasionally input and output files.")

PCFG_UNARY_RULES_DICT_FILE = os.path.join(MODEL_DIR, "PCFG_unary_dict.pkl")
PCFG_BINARY_RULES_DICT_FILE = os.path.join(MODEL_DIR, "PCFG_binary_dict.pkl")
PCFG_POSTAGS_DICT_FILE = os.path.join(MODEL_DIR, "PCFG_postags_dict.pkl")
PCFG_NT_SET_FILE = os.path.join(MODEL_DIR, "PCFG_non_terminals_set.pkl")
PCFG_T_SET_FILE = os.path.join(MODEL_DIR, "PCFG_terminals_set.pkl")
PCFG_POSTAGS_SET_FILE = os.path.join(MODEL_DIR, "PCFG_postags_set.pkl")


# In[ ]:


t0 = time()
print (">>> Loading PCFG model parameters...")

with codecs.open(PCFG_UNARY_RULES_DICT_FILE, 'rb') as f:
    unary_rules_dict = pickle.load(f)
f.close()
with codecs.open(PCFG_BINARY_RULES_DICT_FILE, 'rb') as f:
    binary_rules_dict = pickle.load(f)
f.close()
with codecs.open(PCFG_POSTAGS_DICT_FILE, 'rb') as f:
    postags_dict = pickle.load(f)
f.close()
with codecs.open(PCFG_NT_SET_FILE, 'rb') as f:
    NT_set = pickle.load(f)
f.close()
with codecs.open(PCFG_T_SET_FILE, 'rb') as f:
    T_set = pickle.load(f)
f.close()
with codecs.open(PCFG_POSTAGS_SET_FILE, 'rb') as f:
    postags_set = pickle.load(f)
f.close()

print (">>> PCFG model parameters load done in %0.3fs.\n" % (time() - t0))


# In[ ]:


parser = CYK_parser()
parser.initialize(NT_set, T_set, postags_set, unary_rules_dict, binary_rules_dict, postags_dict)


# In[ ]:


if MODE == "f":
    parser.parse_corpus(input=INPUT_FILE, output=OUTPUT_FILE, verbose=1)
else:
    while True:
        sent = raw_input("Enter the sentence to be parsed (empty string to exit): ").decode('utf8')
        sent = sent.strip()
        if len(sent) == 0:
            break
        else:
            parser.parse_sent(input=sent, verbose=1)

