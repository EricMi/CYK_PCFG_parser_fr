CORPUS_DIR=$1
MODEL_DIR=$2

python corpus_split.py $CORPUS_DIR
python PCFG_learn.py $CORPUS_DIR $MODEL_DIR
python demo.py $MODEL_DIR $3 $4 
