import itertools

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


def tokenizer(sentence):
    return word_tokenize(sentence)


def pos_tagger(tokens):
    return nltk.pos_tag(tokens)


def stopword_treatment(tokens):
    stopword = stopwords.words('english')
    result = []
    for token in tokens:
        if token[0].lower() not in stopword:
            result.append(tuple([token[0].lower(), token[1]]))
    return result


pos_tag_map = {
    'NN': [wn.NOUN],
    'JJ': [wn.ADJ, wn.ADJ_SAT],
    'RB': [wn.ADV],
    'VB': [wn.VERB]
}


def pos_tag_converter(nltk_pos_tag):
    root_tag = nltk_pos_tag[0:2]
    try:
        pos_tag_map[root_tag]
        return pos_tag_map[root_tag]
    except KeyError:
        return ''


def get_synset(token):
    wn_pos_tag = pos_tag_converter(token[1])
    if wn_pos_tag == '':
        return []
    else:
        return wn.synsets(token[0], wn_pos_tag)


def get_tokens_from_synset(synset):
    tokens = {}
    for s in synset:
        if s.name() in tokens:
            tokens[s.name().split('.')[0]] += 1
        else:
            tokens[s.name().split('.')[0]] = 1
    return tokens


def get_hypernyms(token):
    hypernyms = []
    hypernyms.append(token.hypernyms())

    return hypernyms


def get_hyponyms(token):
    hyponyms = []
    hyponyms.append(token.hyponyms())
    return hyponyms


def get_tokens_from_hypernym(synset):
    tokens = []
    for s in synset:
        for ss in s:
            if ss.name().split('.')[0] not in tokens:
                tokens.append(ss.name().split('.')[0])
    return tokens


def underscore_replacer(tokens):
    new_tokens = []
    for token in tokens:
        mod_key = re.sub(r'_', ' ', token)
        new_tokens.append(mod_key)
    return new_tokens


def expand_query(query, score=1.0, n_grams=3):
    tokens = tokenizer(query)
    tokens = pos_tagger(tokens)
    tokens = stopword_treatment(tokens)
    options = {}
    for token in tokens:
        options[token[0]] = [token[0]]
        synset = get_synset(token)
        # print(synset)
        synonyms = get_tokens_from_synset(synset)
        synonyms = underscore_replacer(synonyms.keys())[:3]
        options[token[0]].extend(synonyms)
        # print(f"Synonyms for {token[0]}: {options[token[0]]}")

        if len(synset) > 0:
            hypernyms = get_hypernyms(synset[0])
            hypernyms = get_tokens_from_hypernym(hypernyms)
            hypernyms = underscore_replacer(hypernyms)[:3]
            options[token[0]].extend(hypernyms)
            # print(f"Hypernyms for {token[0]}: {hypernyms}")
            hyponyms = get_hyponyms(synset[0])
            hyponyms = get_tokens_from_hypernym(hyponyms)
            hyponyms = underscore_replacer(hyponyms)[:3]
            options[token[0]].extend(hyponyms)
            # print(f"Hyponyms for {token[0]}: {hyponyms}")

    permutations = list(itertools.product(*list(options.values())))
    permutations = [' '.join(p) for p in permutations]
    permutations = list(set(permutations))
    result = {}
    for perm in permutations:
        original_query = query.split(" ")
        perm_query = perm.split(" ")
        dif = len(set(perm_query) - set(original_query))
        base = min(n_grams, len(perm_query)) + 1
        perm_score = score / (base ** dif)
        # print(f"Query: {perm} | Score: {perm_score}")
        result[perm] = perm_score
    return result


if __name__ == "__main__":
    print(expand_query("michelle obama memoir"))
    print()
    print(expand_query("coronavirus spread"))
    print()
    print(expand_query("bird riding story"))
