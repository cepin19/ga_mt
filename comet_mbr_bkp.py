
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command for Minimum Bayes Risk Decoding.
========================================

This script is inspired in Chantal Amrhein script used in:
    Title: Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET
    URL: https://arxiv.org/abs/2202.05148

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (type: Path_fr, default: null)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (type: Path_fr, default: null)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --num_samples NUM_SAMPLES
                        (required, type: int)
  --model MODEL         COMET model to be used. (type: str, default: wmt20-comet-da)
  --model_storage_path MODEL_STORAGE_PATH
                        Path to the directory where models will be stored. By default its saved in ~/.cache/torch/unbabel_comet/ (default: null)
  -o OUTPUT, --output OUTPUT
                        Best candidates after running MBR decoding. (type: str, default: mbr_result.txt)
"""
import os
from typing import List, Tuple

import numpy as np
import torch
from comet.download_utils import download_model
from comet.models import RegressionMetric, available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from tqdm import tqdm

import sacrebleu
from numpy.random import randint
from numpy.random import rand
import random

from timeit import default_timer as timer
import logging
#from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from comet import download_model, load_from_checkpoint
import sacremoses
tok=sacremoses.MosesTokenizer(lang='en')
detok=sacremoses.MosesDetokenizer(lang='en')
model_path = download_model("wmt21-cometinho-da")
model = load_from_checkpoint(model_path)
model_path = download_model("wmt20-comet-qe-da-v2")
model_qe = load_from_checkpoint(model_path)
emb_cache={}
def build_embeddings(
    sources: List[str],
    translations: List[str],
    model: RegressionMetric,
    batch_size: int,
) -> Tuple[torch.Tensor]:
    """Tokenization and respective encoding of source and translation sentences using
    a RegressionMetric model.

    :param sources: List of source sentences.
    :param translations: List of translation sentences.
    :param model: RegressionMetric model that will be used to embed sentences.
    :param batch_size: batch size used during encoding.

    :return: source and MT embeddings.
    """
    # TODO: Optimize this function to have faster MBR decoding!
    src_batches = [
        sources[i : i + batch_size] for i in range(0, len(sources), batch_size)
    ]
    src_inputs = [model.encoder.prepare_sample(batch) for batch in src_batches]
    mt_batches = [
        translations[i : i + batch_size]
        for i in range(0, len(translations), batch_size)
    ]
    mt_inputs = [model.encoder.prepare_sample(batch) for batch in mt_batches]

    src_embeddings = []
    with torch.no_grad():
        for batch in src_inputs:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            src_embeddings.append(
                model.get_sentence_embedding(input_ids, attention_mask)
            )
    src_embeddings = torch.vstack(src_embeddings)

    mt_embeddings = []
    with torch.no_grad():
        for batch in tqdm(mt_inputs, desc="Encoding sentences...", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            mt_embeddings.append(
                model.get_sentence_embedding(input_ids, attention_mask)
            )
    mt_embeddings = torch.vstack(mt_embeddings)

    return src_embeddings, mt_embeddings

def build_single_embeddings(
    sources: List[str],
    model: RegressionMetric,
    batch_size: int,
        cache,
) -> Tuple[torch.Tensor]:
    """Tokenization and respective encoding of source and translation sentences using
    a RegressionMetric model.

    :param sources: List of source sentences.
    :param translations: List of translation sentences.
    :param model: RegressionMetric model that will be used to embed sentences.
    :param batch_size: batch size used during encoding.

    :return: source and MT embeddings.
    """
    # TODO: Optimize this function to have faster MBR decoding!
    cache_idx=[]
    cache_emb=[]
    new_sources=[]
    #logging.warning(len(sources))
     #logging.warning("cache size: {}".format(len(cache)))

    #logging.warning(cache)

    for i,s in enumerate(sources):
        if s in cache:
            cache_idx.append(i)
            cache_emb.append(cache[s])
        else:
            new_sources.append(s)
    #logging.warning("cache idx: {}".format(cache_idx))
    #logging.warning(len(new_sources))

    if len(new_sources)!=0: #TODO: WHY THIS HAPPENS!!!!!!

        src_batches = [
            new_sources[i : i + batch_size] for i in range(0, len(new_sources), len(new_sources))#batch_size)
        ]
        #logging.warning(batch_size)

        src_inputs = [model.encoder.prepare_sample(batch) for batch in src_batches]
        src_embeddings = []
        with torch.no_grad():
            for batch in src_inputs:
                input_ids = batch["input_ids"].to(model.device)
               # logging.warning(input_ids)
                attention_mask = batch["attention_mask"].to(model.device)
                src_embeddings.append(
                    model.get_sentence_embedding(input_ids, attention_mask)
                )
        src_embeddings = torch.vstack(src_embeddings)
      #  logging.warning(src_embeddings)
     #   logging.warning(src_embeddings.shape)
    final_embs=[]
    cache_i=0
    computed_i=0
    for i in range(len(sources)):
        if i in cache_idx:
            final_embs.append(cache_emb[cache_i])
            cache_i+=1
        else:
      #      logging.warning("appending {}".format(src_embeddings[computed_i]))
            final_embs.append(src_embeddings[computed_i])
            cache[sources[i]]=src_embeddings[computed_i]
            computed_i+=1
    final_embs=torch.vstack(final_embs)
    #logging.warning("final embs:{}".format(final_embs))
    return final_embs

def mbr_decoding(
        src_embeddings: torch.Tensor, mt_embeddings: torch.Tensor,  pseudo_refs: torch.Tensor, model: RegressionMetric, mbr_ref_size: int
) -> torch.Tensor:
    """Performs MBR Decoding for each translation for a given source.

    :param src_embeddings: Embeddings of source sentences.
    :param mt_embeddings: Embeddings of MT sentences.
    :param model: RegressionMetric Model.

    :return:
        Returns a [n_sent x num_samples] matrix M where each line represents a source sentence
        and each column a given sample.
        M[i][j] is the MBR score of sample j for source i.
    """
    n_sent, num_samples, _ = mt_embeddings.shape
    mbr_matrix = torch.zeros(n_sent, num_samples)

    with torch.no_grad():
        # Loop over all source sentences
        for i in tqdm(
            range(mbr_matrix.shape[0]), desc="MBR Scores...", dynamic_ncols=True
        ):
            source = src_embeddings[i, :].repeat(num_samples, 1)
            # Loop over all hypothesis
            for j in range(mbr_matrix.shape[1]):
                #print(j)

                translation = mt_embeddings[i, j, :].repeat(mbr_ref_size, 1)
                # Score current hypothesis against all others

                pseudo_refs = pseudo_refs[:mbr_ref_size]

               # logging.warning(translation.shape)
              #  logging.warning(pseudo_refs.shape)
             #   logging.warning(source.shape)
                scores= model.estimate(source[:mbr_ref_size], translation, pseudo_refs)[
                    "score"
                    ].squeeze(1)
                #logging.warning(scores)

                scores = torch.cat([scores[0:j], scores[j + 1 :]])
                #logging.warning(scores)

                mbr_matrix[i, j] = scores.mean()

    return mbr_matrix



#detok=TreebankWordDetokenizer()
#tok=TreebankWordTokenizer()
n_best=20
def fitness(src,solution,ref):
#    print(' '.join(solution))
    s=sacrebleu.sentence_bleu(detok.detokenize(solution).strip() , [ref], smooth_method='exp')

#s=sacrebleu.sentence_chrf(' '.join(solution).strip() , ["Toto je asfs test."])

    return s.score
def fitness_comet_single(src,solution,ref):
    data=[
        {
                "src": src,
                "mt": detok.detokenize(solution).strip(),
                "ref": ref
        }]
    seg_scores, sys_score = model.predict(data, batch_size=1, gpus=1)
    return sys_score

def fitness_comet_multi(src,solutions,ref):
    solutions=[ list(filter(lambda t: t != '', solution)) for solution in solutions]
    data=[
            {
            "src": src,
             "mt": detok.detokenize(solution).strip()#,
#            "ref": ref
           } for solution in solutions]
    seg_scores, sys_score = model.predict(data, batch_size=len(solutions)//4, gpus=4, progress_bar=False)
    return seg_scores

def fitness_comet_mbr(src_embeddings,solutions, ref, init_mt_embeddings=None):
    #TODO: USE INITIAL TRANSLATIONS AS PSEUODOREFS (complexity)
    solutions=[detok.detokenize(solution) for solution in solutions if solution!='']
   # print(solutions)
   # print(len(solutions))
    start = timer()

    mt_embeddings=build_single_embeddings(solutions,model,len(solutions),emb_cache)
    end = timer()
    logging.info("Computing embeddings for the solutions took {} s".format(end-start))
    mt_embeddings = mt_embeddings.reshape(1, len(solutions), -1)

    return mbr_decoding(src_embeddings, mt_embeddings, init_mt_embeddings, model, mbr_ref_size=20)[0] # only one source, so [0]

def fitness_comet_qe(src,solutions):
    solutions=[ list(filter(lambda t: t != '', solution)) for solution in solutions]
    data=[
            {
            "src": src,
             "mt": detok.detokenize(solution).strip()#,
#            "ref": ref
           } for solution in solutions]
    seg_scores, sys_score = model_qe.predict(data, 64, gpus=1, progress_bar=False)
    return seg_scores
def fitness_comet_mbr(src,src_embeddings,solutions, ref, init_mt_embeddings=None):
    #TODO: USE INITIAL TRANSLATIONS AS PSEUODOREFS (complexity)
    solutions=[detok.detokenize(solution) for solution in solutions if solution!='']
   # print(solutions)
   # print(len(solutions))
    start = timer()

    mt_embeddings=build_single_embeddings(solutions,model,len(solutions),emb_cache)
    end = timer()
    logging.info("Computing embeddings for the solutions took {} s".format(end-start))
    mt_embeddings = mt_embeddings.reshape(1, len(solutions), -1)

    return mbr_decoding(src_embeddings, mt_embeddings, init_mt_embeddings, model, mbr_ref_size=20)[0] # only one source, so [0]


def fitness_comet_mbr_and_qe(src,src_embeddings,solutions,pseudo_ref,init_mt_embeddings):
    start = timer()
    mbr_scores=fitness_comet_mbr(src,src_embeddings,solutions,pseudo_ref,init_mt_embeddings=init_mt_embeddings)
    end = timer()
    logging.info("Computing MBR scores took {} s".format(end-start))
    start = timer()
    qe_scores=np.asarray(fitness_comet_qe(src, solutions))
    end = timer()
    logging.info("Computing QE scores took {} s".format(end-start))
    #logging.warning(mbr_scores.shape)
    #logging.warning(qe_scores.shape)
    #logging.warning(mbr_scores.shape)
    #logging.warning(qe_scores)
    #logging.warning(qe_scores.shape)

    return mbr_scores+qe_scores

def mutation(bitstring, r_mut, possible_tgt):
    #TODO: solve for multi-token expressions
    #It should be more probable to replace existing word rather than an emtpy ony
    # TODO: solve for multi-token expressions
    # It should be more probable to replace existing word rather than an emtpy ony
    empty_repl = 0.1
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            if bitstring[i] == '':
                if rand() > empty_repl:
                    continue
            tgt = random.choice(possible_tgt)  # .split(' ')
            l = len(tgt)
            if l > 1:
                # find empty places in the gene
                space_i = [a for a, x in enumerate(bitstring) if x == '']
                if l > len(space_i):  # wait, thats illegal (we cant make the gene longer)
                    continue
                #            for x in range(l):
                # print(bitstring)
                #               print(space_i)
                #              print(x)
                #             print(space_i[-x-1])
                #
                #                   del bitstring[space_i[-x-1]]  #we need to iterate backwards to not mess up the order
                #              print(i)
                #             print(bitstring)
                #                logging.warning("inserting {} instead of {}".format(tgt, bitstring[i]))
                if bitstring[i]:
                    if i==0 and bitstring[i][0].isupper(): #uppercase the first letter, if start of the sentece
                        tgt[0]=tgt[0].capitalize()
                bitstring[i] = tgt[0]
                for x in range(1, l):
                    bitstring.insert(i + x, tgt[x])
                    space_i = [a for a, t in enumerate(bitstring) if t == '']
                    # logging.warning(space_i)
                    # logging.warning(x)
                    # logging.warning(bitstring)
                    del bitstring[space_i[-1]]
            else:
                x = 0
                bitstring[i] = tgt[x]
# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
#        print("Crossing over")
 #       print(c1)
  #      print(c2)
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
#        pt=2
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
   #     print("new crossover:")
    #    print(c1)
     #   print(c2)
    return [c1, c2]

# genetic algorithm
def genetic_algorithm(objective, init_pop, n_bits, n_iter, n_pop, r_cross, r_mut,src,ref,possible_tgt,src_embeddings=None,init_mt_embeddings=None):
    # initial population of random bitstring
    # keep track of best solution

    pop=init_pop
    if src_embeddings is not None:
        best, best_eval = pop[0], objective(src,src_embeddings,pop,ref,init_mt_embeddings=init_mt_embeddings)[0]
    else:
        best, best_eval = pop[0], objective(src,pop,ref)[0]

    logging.warning("initial best: {}".format(best))
    logging.warning("initial best fitness: {}".format(best_eval))
    #logging.warning("tgt words: {}".format(possible_tgt))
    # enumerate generations
    for gen in tqdm(range(n_iter),desc='generation'):
        # evaluate all candidates in the population
#        scores = [objective(src,c,ref) for c in pop]
        if src_embeddings is not None:
            scores=objective(src,src_embeddings,pop,ref,init_mt_embeddings=init_mt_embeddings)
        else:
            scores=objective(src,pop,ref)
        logging.warning("avg fitness: {}".format(sum(scores)/len(scores)))

        #for p, s in zip(pop,scores):
        #    logging.warning("{} = {}".format(detok.detokenize(p).strip(),s))
        # check for new best solution


        #logging.warning("scores:{}".format(scores))
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                logging.warning("Iteration %d: >%d, new best f(%s) = %.3f" % (gen, gen,  pop[i], scores[i]))
        # select parents
        #logging.info ("before:")
        #for p in pop:
         #   logging.info (' '.join(p))
        #logging.info("Len: {}".format(len(pop)))
        #logging.info("n_pop: {}".format(n_pop))

        selected = [selection(pop, scores) for _ in range(n_pop)]

        #logging.info ("after:")
        #for p in selected:
         #   logging.info (' '.join(p))
        #logging.info("Len: {}".format(len(selected)))
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut,possible_tgt)
                # store for next generation
#                print(c)
                children.append(c)
        # replace population
        pop = children
        if gen % 10 == 0:
            logging.warning("gen: {}".format(gen))
            logging.warning("pop sample: {}".format('\n'.join(detok.detokenize(p) for p in pop[:250])))

    return [best, best_eval]

def ga_init(cfg):
    n_best=20
    #with open("{}.trans".format(f)) as trans, open("{}.scores".format(f)) as scores, open(
     #       "{}.ref".format(f)) as refs, open("{}.src".format(f)) as srcs, open(
      #      "{}.tgt_words_exp".format(f)) as dict_tgts:
    with open(cfg.sources()) as srcs, open(cfg.translations()) as trans, open(cfg.ref()) as refs, open(cfg.dict()) as dict_tgts:
        translations = trans.readlines()
        i = 0
        for dict_tgt, src, ref in zip(dict_tgts, srcs, refs):
            translations_sent = translations[i * n_best:(i + 1) * n_best]
            init_pop = ([tok.tokenize(s) for s in translations_sent])
            # print(init_pop)
            pop = []
            # refactor
            for s in init_pop:
                news = []
                for t in s:
                    news.append(t)
                    news.append('')
                pop.append(news)
            # possible_tgt=["Toto", "je", "test","asfs",""]
          #  possible_tgt = list(set([t for sent in init_pop for t in sent])) + [''] + dict_tgt.strip().split(' ')
            tgt_toks = list(set([tok for sent in init_pop for tok in sent]))
            tgt_toks = [[tok] for tok in tgt_toks]
            dict_toks = [t.split(' ') for t in dict_tgt.strip().split(';')]
            possible_tgt = tgt_toks + [['']]*(len(tgt_toks)+len(dict_toks)) + dict_toks

            logging.info(possible_tgt)
            #exit()
            max_len = int(max([len(s) for s in pop]) * 1.1)
            # PAD
            pop = [(p + max_len * [''])[:max_len] for p in pop] * 100
            # print(pop)
            #        print(pop

            #print(src)
            src_embeddings= build_single_embeddings([src], model, len(src),emb_cache)
            init_mt_embeddings=build_single_embeddings(translations_sent, model, len(src),emb_cache)

            out = genetic_algorithm(fitness_comet_mbr_and_qe, pop, max_len, cfg.generations, len(pop), cfg.crossover, cfg.mutation, src, ref, possible_tgt,src_embeddings=src_embeddings,init_mt_embeddings=init_mt_embeddings)[0]
            out = list(filter(lambda t: t != '', out))

            print(detok.detokenize(out))
            i += 1
    #        if i>1:
    # break
def mbr_command() -> None:
    parser = ArgumentParser(description="Command for Minimum Bayes Risk Decoding.")
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-t", "--translations", type=Path_fr)
    parser.add_argument("-r", "--ref", type=Path_fr)
    parser.add_argument("-m", "--mutation", type=float, default=0.05)
    parser.add_argument("-c", "--crossover", type=float, default=0.05)
    parser.add_argument("-g", "--generations", type=int, default=100)

    parser.add_argument("-d", "--dict", type=Path_fr)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="wmt20-comet-da",
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--model_storage_path",
        help=(
            "Path to the directory where models will be stored. "
            + "By default its saved in ~/.cache/torch/unbabel_comet/"
        ),
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mbr_result.txt",
        help="Best candidates after running MBR decoding.",
    )
    cfg = parser.parse_args()

    if cfg.sources is None:
        parser.error("You must specify a source (-s)")

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    elif cfg.model in available_metrics:

        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)
    else:
        parser.error(
            "{} is not a valid checkpoint path or model choice. Choose from {}".format(
                cfg.model, available_metrics.keys()
            )
        )
    model = load_from_checkpoint(model_path)

    if not isinstance(model, RegressionMetric):
        raise Exception(
            "Incorrect model ({}). MBR command only works with RegressionMetric models.".format(
                model.__class__.__name__
            )
        )

    model.eval()
    model.cuda()

    with open(cfg.sources()) as fp:
        sources = [line.strip() for line in fp.readlines()]

    with open(cfg.translations()) as fp:
        translations = [line.strip() for line in fp.readlines()]

    ga_init(cfg)
    exit()
    src_embeddings, mt_embeddings = build_embeddings(
        sources, translations, model, cfg.batch_size
    )
    mt_embeddings = mt_embeddings.reshape(len(sources), cfg.num_samples, -1)
    mbr_matrix = mbr_decoding(src_embeddings, mt_embeddings, model)
    translations = [
        translations[i : i + cfg.num_samples]
        for i in range(0, len(translations), cfg.num_samples)
    ]
    assert len(sources) == len(translations)

    best_candidates = []
    for i, samples in enumerate(translations):
        best_cand_idx = torch.argmax(mbr_matrix[i, :])
        best_candidates.append(samples[best_cand_idx])

    with open(cfg.output, "w") as fp:
        for sample in best_candidates:
            fp.write(sample + "\n")


if __name__ == "__main__":
    mbr_command()

