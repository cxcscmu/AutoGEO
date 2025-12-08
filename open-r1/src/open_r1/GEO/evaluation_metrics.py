import math
import itertools
from glob import glob

PROMPT_TEMPLATE = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg}[/INST]"

import re
import nltk

def get_num_words(line):
    return len([x for x in line if len(x)>2])

def extract_citations_new(text):
    def ecn(sentence):
        citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
        return [int(re.findall(r'\d+', citation)[0]) for citation in re.findall(citation_pattern, sentence)]
    paras = re.split(r'\n\n', text)
    sentences = [nltk.sent_tokenize(p) for p in paras]
    words = [[(nltk.word_tokenize(s), s, ecn(s)) for s in sentence] for sentence in sentences]
    return words

def impression_wordpos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])

            try: scores[cit] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores

def impression_word_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try: scores[cit] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
    

def impression_pos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = 1
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: scores[cit] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
