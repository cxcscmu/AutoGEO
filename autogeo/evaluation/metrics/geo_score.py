import math
import itertools
import re
import nltk

def get_num_words(line: list) -> int:
    """Count number of words in a tokenized line (words longer than 2 characters).
    
    Args:
        line: List of word tokens
        
    Returns:
        Number of words longer than 2 characters
    """
    return len([x for x in line if len(x)>2])

def extract_citations_new(text: str) -> list:
    """Extract citations from text and organize by paragraphs and sentences.
    
    Args:
        text: Text containing citations in [n] format
        
    Returns:
        Nested list structure: paragraphs -> sentences -> (tokens, sentence_text, citation_indices)
    """
    def ecn(sentence):
        citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'

        return [int(re.findall(r'\d+', citation)[0]) for citation in re.findall(citation_pattern, sentence)]

    paras = re.split(r'\n\n', text)
    sentences = [nltk.sent_tokenize(p) for p in paras]
    words = [[(nltk.word_tokenize(s), s, ecn(s)) for s in sentence] for sentence in sentences]
    return words

def impression_wordpos_count_simple(sentences: list, n: int = 5, normalize: bool = True) -> list:
    """Calculate impression score based on word count and position.
    
    Args:
        sentences: Nested list of sentences with citations
        n: Number of documents to score (default: 5)
        normalize: Whether to normalize scores (default: True)
        
    Returns:
        List of scores for each document
    """
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: 
                scores[cit] += score
            except: 
                # print(f'Citation Hallucinated: {cit}')
                pass
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores

def impression_word_count_simple(sentences: list, n: int = 5, normalize: bool = True) -> list:
    """Calculate impression score based on word count only.
    
    Args:
        sentences: Nested list of sentences with citations
        n: Number of documents to score (default: 5)
        normalize: Whether to normalize scores (default: True)
        
    Returns:
        List of scores for each document
    """
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try: scores[cit] += score
            except: # print(f'Citation Hallucinated: {cit}')
                pass
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
    

def impression_pos_count_simple(sentences: list, n: int = 5, normalize: bool = True) -> list:
    """Calculate impression score based on position only.
    
    Args:
        sentences: Nested list of sentences with citations
        n: Number of documents to score (default: 5)
        normalize: Whether to normalize scores (default: True)
        
    Returns:
        List of scores for each document
    """
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = 1
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: scores[cit] += score
            except: # print(f'Citation Hallucinated: {cit}')
                pass
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
