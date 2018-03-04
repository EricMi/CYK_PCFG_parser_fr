
# coding: utf-8

# In[1]:




import os
import sys
import codecs
import random
from time import time
import string
import re
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

CORPUS_DIR = sys.argv[1]
CORPUS = os.path.join(CORPUS_DIR, "sequoia-corpus+fct.mrg_strict")
TB_TRAIN = os.path.join(CORPUS_DIR, "sequoia_train.tb")
TB_DEV = os.path.join(CORPUS_DIR, "sequoia_dev.tb")
TB_TEST = os.path.join(CORPUS_DIR, "sequoia_test.tb")
TXT_DEV = os.path.join(CORPUS_DIR, "sequoia_dev.txt")
TXT_TEST = os.path.join(CORPUS_DIR, "sequoia_test.txt")

t0 = time()
print (">>> Splitting corpus into train/dev/test sets...")

nt_funcl_re = re.compile(r"(?<=\()[A-Za-z_+^\-]+\-[^ ]+")
def remove_funcl(m):
    return m.group().split('-')[0]

f_in = codecs.open(CORPUS, 'r', 'UTF-8')
data = f_in.read().splitlines()
for i in range(len(data)):
    data[i] = nt_funcl_re.sub(lambda x: remove_funcl(x), data[i])
f_in.close()

s_total = len(data)
p_train = 0.8
p_dev = 0.1
p_test = 0.1

random.seed(39)
random.shuffle(data)

corpus_train = data[:int(s_total * p_train)]
f_train = codecs.open(TB_TRAIN, 'w', 'UTF-8')
for s in corpus_train:
    f_train.write(u"{0}\n".format(s))
f_train.close()

corpus_dev = data[int(s_total * p_train) : int(s_total * (p_train + p_dev))]
f_dev = codecs.open(TB_DEV, 'w', 'UTF-8')
for s in corpus_dev:
    f_dev.write(u"{0}\n".format(s))
f_dev.close()

corpus_test = data[int(s_total * (p_train + p_dev)):]
f_test = codecs.open(TB_TEST, 'w', 'UTF-8')
for s in corpus_test:
    f_test.write(u"{0}\n".format(s))
f_test.close()

corpus_root = r"./corpus/"

dev_file_pattern = r".*_dev\.tb"
ptb_dev = BracketParseCorpusReader(corpus_root, dev_file_pattern)
trees = ptb_dev.parsed_sents()
f_out = codecs.open(TXT_DEV, 'w', 'UTF-8')
for tree in trees:
    f_out.write(u"{0}\n".format(u" ".join(tree.leaves())))
f_out.close()

test_file_pattern = r".*_test\.tb"
ptb_test = BracketParseCorpusReader(corpus_root, test_file_pattern)
trees = ptb_test.parsed_sents()
f_out = codecs.open(TXT_TEST, 'w', 'UTF-8')
for tree in trees:
    f_out.write(u"{0}\n".format(u" ".join(tree.leaves())))
f_out.close()

print ("Corpus size: %d" % s_total)
print ("Train set size: %d" % len(corpus_train))
print ("Dev set size: %d" % len(corpus_dev))
print ("Test set size: %d" % len(corpus_test))

print (">>> Corpus split done in %0.3fs.\n" % (time() - t0))

