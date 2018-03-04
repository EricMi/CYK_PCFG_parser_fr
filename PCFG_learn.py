
# coding: utf-8

# In[1]:




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
from collections import defaultdict
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

MODEL_DIR = sys.argv[2]

PCFG_UNARY_RULES_FREQ_FILE = os.path.join(MODEL_DIR, "PCFG_unary_freq.pkl")
PCFG_BINARY_RULES_FREQ_FILE = os.path.join(MODEL_DIR, "PCFG_binary_freq.pkl")
PCFG_POSTAGS_FREQ_FILE = os.path.join(MODEL_DIR, "PCFG_postags_freq.pkl")
PCFG_UNARY_RULES_DICT_FILE = os.path.join(MODEL_DIR, "PCFG_unary_dict.pkl")
PCFG_BINARY_RULES_DICT_FILE = os.path.join(MODEL_DIR, "PCFG_binary_dict.pkl")
PCFG_POSTAGS_DICT_FILE = os.path.join(MODEL_DIR, "PCFG_postags_dict.pkl")
PCFG_NT_SET_FILE = os.path.join(MODEL_DIR, "PCFG_non_terminals_set.pkl")
PCFG_T_SET_FILE = os.path.join(MODEL_DIR, "PCFG_terminals_set.pkl")
PCFG_POSTAGS_SET_FILE = os.path.join(MODEL_DIR, "PCFG_postags_set.pkl")


# In[2]:


t0 = time()
print (">>> Reading corpus treebanks from file...")

corpus_root = sys.argv[1]
train_file_pattern = r".*_train.tb"

ptb_train = BracketParseCorpusReader(corpus_root, train_file_pattern)

print (">>> Corpus treebanks read done in %0.3fs.\n" % (time() - t0))


# In[3]:


t0 = time()
print (">>> Parsing collection of rules and words...")

# Objects for unary rules (A -> B)
unary_rules_freq = defaultdict(float)
unary_rules_cnt_by_lhs = defaultdict(int)
unary_rules_occur_cnt = 0
unary_lhs_set = set()
unary_rhs_set = set()

# Objects for binary rules (A -> BC)
binary_rules_freq = defaultdict(float)
binary_rules_cnt_by_lhs = defaultdict(int)
binary_rules_occur_cnt = 0
binary_lhs_set = set()
binary_rhs_set = set()

# Objects for terminal rules (POS -> <word>)
postags_freq = defaultdict(float)
postags_cnt_by_pos = defaultdict(int)
postags_occur_cnt = 0
words_occur_cnt = defaultdict(int)
postags_set = set()
words_set = set()

trees = ptb_train.parsed_sents()
for tree in trees:
    t = tree.copy()
    t.chomsky_normal_form(horzMarkov=2)
    #t.collapse_unary(collapsePOS=True, collapseRoot=False)
    prods = t.productions()
    for prod in prods:
        lhs = prod.lhs().symbol()
        rh = prod.rhs()
        #rhs = ' '.join([r.symbol() if isinstance(r, nltk.grammar.Nonterminal) else r for r in rh])
        if isinstance(rh[0], unicode): # Ternimal production (POS -> <word>)
            rhs = rh[0]
            postags_freq[(lhs, rhs)] += 1
            postags_cnt_by_pos[lhs] += 1
            postags_occur_cnt += 1
            words_occur_cnt[rhs] += 1
            postags_set.add(lhs)
            words_set.add(rhs)
        else: # Non-terminal production (A -> BC | A -> B)
            if len(rh) == 1: # Unary production (A -> B)    
                rhs = rh[0].symbol()
                unary_rules_freq[(lhs, rhs)] += 1
                unary_rules_cnt_by_lhs[lhs] += 1
                unary_rules_occur_cnt += 1
                unary_lhs_set.add(lhs)
                unary_rhs_set.add(rhs)
            elif len(rh) == 2:
                rhs = tuple([nt.symbol() for nt in rh])
                binary_rules_freq[(lhs, rhs)] += 1
                binary_rules_cnt_by_lhs[lhs] += 1
                binary_rules_occur_cnt += 1
                binary_lhs_set.add(lhs)
                binary_rhs_set.add(rhs)

# Replace rare words in the postags_freq with '<UNK>'
rare_words = set([w for w in words_set if words_occur_cnt[w] < 2])
T_set = words_set.copy()
T_set.difference_update(rare_words)
T_set.add(u"<UNK>")

pw_pairs = list(postags_freq.keys())
for (pos, w) in pw_pairs:
    if w in rare_words:
        postags_freq[(pos, u"<UNK>")] += postags_freq[(pos, w)]
        postags_freq.pop((pos, w))

# Normalization to ensure that the sum of outgoing weights is 1 for each NT node
for (pos, w) in postags_freq:
    postags_freq[(pos, w)] /= postags_cnt_by_pos[pos]

for (lhs, rhs) in unary_rules_freq:
    unary_rules_freq[(lhs, rhs)] /= (unary_rules_cnt_by_lhs[lhs] + binary_rules_cnt_by_lhs[lhs])
    
for (lhs, rhs) in binary_rules_freq:
    binary_rules_freq[(lhs, rhs)] /= (binary_rules_cnt_by_lhs[lhs] + unary_rules_cnt_by_lhs[lhs])

with codecs.open(PCFG_UNARY_RULES_FREQ_FILE, 'wb') as f:
    pickle.dump(unary_rules_freq, f)
f.close()

with codecs.open(PCFG_BINARY_RULES_FREQ_FILE, 'wb') as f:
    pickle.dump(binary_rules_freq, f)
f.close()

with codecs.open(PCFG_POSTAGS_FREQ_FILE, 'wb') as f:
    pickle.dump(postags_freq, f)
f.close()

# Construct the rhs -> lhs dictionary for quick parent lookup in CYK algorithm
unary_rules_dict = {}
binary_rules_dict = {}
postags_dict = {}

for rhs in unary_rhs_set:
    unary_rules_dict[rhs] = {}
for (lhs, rhs) in unary_rules_freq:
    unary_rules_dict[rhs][lhs] = unary_rules_freq[(lhs, rhs)]
    
for rhs in binary_rhs_set:
    binary_rules_dict[rhs] = {}
for (lhs, rhs) in binary_rules_freq:
    binary_rules_dict[rhs][lhs] = binary_rules_freq[(lhs, rhs)]
    
for w in T_set:
    postags_dict[w] = {}
for (pos, w) in postags_freq:
    postags_dict[w][pos] = postags_freq[(pos, w)]

with codecs.open(PCFG_UNARY_RULES_DICT_FILE, 'wb') as f:
    pickle.dump(unary_rules_dict, f)
f.close()

with codecs.open(PCFG_BINARY_RULES_DICT_FILE, 'wb') as f:
    pickle.dump(binary_rules_dict, f)
f.close()

with codecs.open(PCFG_POSTAGS_DICT_FILE, 'wb') as f:
    pickle.dump(postags_dict, f)
f.close()

# Store the set of non-terminals and terminals
NT_set = unary_lhs_set.union(binary_lhs_set)

with codecs.open(PCFG_NT_SET_FILE, 'wb') as f:
    pickle.dump(NT_set, f)
f.close()

with codecs.open(PCFG_T_SET_FILE, 'wb') as f:
    pickle.dump(T_set, f)
f.close()

with codecs.open(PCFG_POSTAGS_SET_FILE, 'wb') as f:
    pickle.dump(postags_set, f)
f.close()
        
print ("Size of dictionary: %d, of which %d are rare words" % (len(words_set), len(rare_words)))
print ("Number of word occurrances: %d" % postags_occur_cnt)
print ("Number of POS tags: %d\n" % len(postags_set))

print ("Size of unary rules: %d" % len(unary_rules_freq))
print ("Number of unary rule occurrances: %d" % unary_rules_occur_cnt)
print ("Number of unary lhs: %d" % len(unary_lhs_set))
print ("Number of unary rhs: %d\n" % len(unary_rhs_set))

print ("Size of binary rules: %d" % len(binary_rules_freq))
print ("Number of binary rule occurrances: %d" % binary_rules_occur_cnt)
print ("Number of binary lhs: %d" % len(binary_lhs_set))
print ("Number of binary rhs: %d\n" % len(binary_rhs_set))

print ("Number of non-terminal symbols: %d" % len(NT_set))
print ("Number of terminal symbols: %d\n" % len(T_set))

print (">>> Collection of rules and words parsing done in %0.3fs.\n" % (time() - t0))

