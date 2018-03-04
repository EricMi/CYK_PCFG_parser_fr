# CYK_PCFG_parser_fr
A basic probabilistic parser for French based on the CYK algorithm and the PCFG model.


## Requirements

Python 2.7
nltk 3.2.5


## Quick start

The "demo" script provides a tutorial of how to load pre-trained PCFG model and
parse sentences with the CYK_parser class.

To learn the steps of training PCFG model on your own corpus, please refer to
the run_parser.sh as following:

```shell
$ ./run_parser.sh CORPUS_DIR MODEL_DIR [INPUT_FILE OUTPUT_FILE]
```

* CORPUS_DIR: the directory of training corpus
* MODEL_DIR: the direcroty to store PCFG parameters
* INPUT_FILE: the text file to parse, one sentence per line as whitespace
  sperated tokens
* OUTPUT_FILE: the file to store parsed structures
*NOTE* If yout omit the last two parameters, you will enter interactive mode,
where you can enter one sentence at a time in console, and get parse reault in
console.


### Example

```shell
$ ./run_parser.sh corpus model corpus/demo.txt demo.out
```

This command will use the "corpus/sequoia-corpus+fct.mrg_strict" as training
corpus.

Firstly, it will be splitted into train/dev/test sets.

Secondly, the PCFG_learn script will extract PCFG frequencies and store them in
"model" folder.

Finally, it parse the sentences in "corpus/demo.txt", then store the parse
result in "demo.out"


## run_parser.sh

```shell
CORPUS_DIR=$1
MODEL_DIR=$2

python corpus_split.py $CORPUS_DIR
python PCFG_learn.py $CORPUS_DIR $MODEL_DIR
python demo.py $MODEL_DIR $3 $4 
```

## References

[1] Jurafsky, Dan, and James H. Martin. Speech and language processing. Vol. 3. London:: Pearson, 2014.
[2] Chappelier, Jean-Cédric, and Martin Rajman. "A Generalized CYK Algorithm for Parsing Stochastic CFG." TAPD 98.133-137 (1998): 5.

