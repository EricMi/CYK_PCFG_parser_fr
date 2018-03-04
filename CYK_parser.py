#-*- coding: utf-8 -*-

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


"""
# Check freq normalization
lhs_set = NT_set.copy().union(postags_set)
freq_sum_by_lhs = defaultdict(float)

for (lhs, rhs) in unary_rules_freq.iterkeys():
    freq_sum_by_lhs[lhs] += unary_rules_freq[(lhs, rhs)]
for (lhs, rhs) in binary_rules_freq.iterkeys():
    freq_sum_by_lhs[lhs] += binary_rules_freq[(lhs, rhs)]
for (pos, w) in postags_freq:
    freq_sum_by_lhs[pos] += postags_freq[(pos, w)]

flag = True
for v in freq_sum_by_lhs.itervalues():
    if abs(v - 1) > 1e-10:
        flag = False
        break

if flag:
    print ("PCFG model well normalized!")
else:
    print ("PCFG model not well normalized!")
"""

class CYK_parser(object):
    def __init__(self):
        self.NT_set = set()                            # set of non-terminal symbols
        self.T_set = set()                             # set of terminal symbols
        self.postags_set = set()                       # set of postags
        #self.unary_rules_freq = defaultdict(float)     # frequencies of unary rules (A -> B)
        #self.binary_rules_freq = defaultdict(float)    # frequencies of binary rules (A -> BC)
        #self.postags_freq = defaultdict(float)         # frequencies of postags (POS -> <word>)
        self.unary_rules_dict = {}
        self.binary_rules_dict = {}
        self.postags_dict = {}
        self.not_initialized = True

    def initialize(self, NT_set, T_set, postags_set,
                   #unary_rules_freq, binary_rules_freq, postags_freq,
                   unary_rules_dict, binary_rules_dict, postags_dict):
        self.NT_set = NT_set
        self.T_set = T_set
        self.postags_set = postags_set
        #self.unary_rules_freq = unary_rules_freq
        #self.binary_rules_freq = binary_rules_freq
        #self.postags_freq = postags_freq
        self.unary_rules_dict = unary_rules_dict
        self.binary_rules_dict = binary_rules_dict
        self.postags_dict = postags_dict
        self.not_initialized = False
        
    def _parse_sent(self, s, verbose=False):
        t0 = time()
        
        tokens = s.strip().split(u' ')
        n = len(tokens)
        dp = defaultdict(float)
        backPointers = {}
        
        # POS tagger
        for i, w in enumerate(tokens):
            if w in self.T_set:
                dp[(i, i+1)] = self.postags_dict[w]
            else:
                dp[(i, i+1)] = self.postags_dict[u"<UNK>"]
            if verbose > 1:
                print (u"->Add POS tag for {0}:\n".format(w))
                print (dp[(i, i+1)])
            self.add_unary_rules(dp, backPointers, i, i+1, verbose)
        
        for l in range(2, n + 1):
            for i in range(0, n + 1 - l):
                j = i + l
                dp[(i, j)] = {}
                for s in range(i + 1, j):
                    B_set = dp[(i, s)]
                    C_set = dp[(s, j)]
                    for B, prob_B in B_set.iteritems():
                        for C, prob_C in C_set.iteritems():
                            if (B, C) in self.binary_rules_dict:
                                for A, prob_A in self.binary_rules_dict[(B, C)].iteritems():
                                    prob = prob_A * prob_B * prob_C
                                    if (A not in dp[(i, j)]) or prob > dp[(i, j)][A]:
                                        dp[(i, j)][A] = prob
                                        backPointers[(i, j, A)] = (s, B, C)
                                        if verbose > 1:
                                            print (u"-->Add binary rule ({0}, {1}): {2} -> {3} {4} / {5}\n".format(i, j, A, B, C, prob))
                self.add_unary_rules(dp, backPointers, i, j, verbose)
        
        if (0, n, u"SENT") not in backPointers:
            return None
        else:
            t = self.buildTree(backPointers, 0, n, u"SENT", tokens)
            t.un_chomsky_normal_form(expandUnary = False)
            return t
 
    def add_unary_rules(self, dp, backPointers, i, j, verbose=False):
        B_set = dp[(i, j)].keys()
        for B in B_set:
            if B in self.unary_rules_dict:
                for A, prob_A in self.unary_rules_dict[B].iteritems():
                    prob = prob_A * dp[(i, j)][B]
                    if (A not in dp[(i, j)]) or prob > dp[(i, j)][A]:
                        dp[(i, j)][A] = prob
                        backPointers[(i, j, A)] = (B,)
                        if verbose > 1:
                            print (u"-->Add unary rule ({0}, {1}): {2} -> {3} / {4}\n".format(i, j, A, B, prob))
        return
    
    def buildTree(self, backPointers, i, j, label, tokens):
        if (i, j, label) not in backPointers: # Terminals
            t = Tree(label, [tokens[i]])
        elif len(backPointers[(i, j, label)]) == 1: # Unary rules
            child_label = backPointers[(i, j, label)][0]
            t = Tree(label, [self.buildTree(backPointers, i, j, child_label, tokens)])
        else: # Binary rules
            split, child_label0, child_label1 = backPointers[(i, j, label)]
            t = Tree(label, [self.buildTree(backPointers, i, split, child_label0, tokens),
                             self.buildTree(backPointers, split, j, child_label1, tokens)])
        return t
    
    def parse_sent(self, input, output=None, verbose=False):
        if self.not_initialized:
            print ("Parser must be initialized before calling parse function!")
            return
        
        t0 = time()
        
        tree = self._parse_sent(input, verbose)
        if output == None:
            print (tree)
        else:
            with codecs.open(output, 'w', 'UTF-8') as f:
                f.write(u"( {0})".format(u" ".join(str(tree).split())))
                f.close()
                
        if verbose:
            print ("Sentence parse done in %0.3fs" % (time() - t0))
        return tree
        
        
    def parse_corpus(self, input, output=None, verbose=False):
        if self.not_initialized:
            print ("Parser must be initialized before calling parse function!")
            return
        
        to = time()
        
        with codecs.open(input, 'r', 'UTF-8') as f_in:
            if output != None:
                f_out = codecs.open(output, 'w', 'UTF-8')
            data = f_in.read().splitlines()
            n_sents = len(data)
            for i, sent in enumerate(data):
                tree = self._parse_sent(sent, verbose)
                if output != None:
                    f_out.write(u"( {0})".format(u" ".join(str(tree).split())))
                else:
                    print (u"( {0})".format(u" ".join(str(tree).split())))
                if verbose:
                    print ("{0}/{1} sentences done...".format(i + 1, n_sents))
            f_in.close()
            if output != None:
                f_out.close()
                
        if verbose:
            print ("Corpus parse done in %0.3fs" % (time() - t0))

