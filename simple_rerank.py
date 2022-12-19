from numpy.random import randint
from numpy.random import rand
import random
import nltk,sys
import logging
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from comet import download_model, load_from_checkpoint

model_path = download_model("wmt20-comet-da")
model = load_from_checkpoint(model_path)



detok=TreebankWordDetokenizer()
tok=TreebankWordTokenizer()
n_best=int(sys.argv[4])

def score_comet_multi(src,solutions,ref):
    data=[
            {
            "src": src,
             "mt": detok.detokenize(solution).strip(),
            "ref": ref
           } for solution in solutions]
    seg_scores, sys_score = model.predict(data, batch_size=len(solutions), gpus=1, progress_bar=False)
    return seg_scores


srcf=sys.argv[1]
tgtf=sys.argv[2]
reff=sys.argv[3]

with open(tgtf) as trans,  open(reff) as refs, open(srcf) as srcs:
    translations=trans.readlines()
    i=0
    for src,ref in zip(srcs,refs):
        translations_sent=translations[i*n_best:(i+1)*n_best]
        pop=([nltk.word_tokenize(s) for s in translations_sent])
        scores=score_comet_multi(src,pop,ref)
        best=scores.index(max(scores))
        print(detok.detokenize(pop[best]).strip())
        i+=1

