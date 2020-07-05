from collections import Counter

class BagOfWords:
    def __init__(self):
        pass

    def calc_bow(self, tokenized_texts):
        # Build vocabulary
        vocabulary = {}
        for tokenized_text in tokenized_texts:
            for token in tokenized_text:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        n_vocab = len(vocabulary)
        # Build Bag of Words Feature Vector
        bow = [[0] * n_vocab for i in range(len(tokenized_texts))]
        for i, tokenized_text in enumerate(tokenized_texts):
            for token in tokenized_text:
                index = vocabulary[token]
                bow[i][index] += 1
        return vocabulary, bow

    def calc_bow_counter_ver(self, tokenized_texts):
        # Build vocabulary
        counts = [Counter(tokenized_text) for tokenized_text in tokenized_texts]
        sum_counts = sum(counts, Counter())
        vocabulary = sum_counts.keys()
        # Build Bag of Words Feature Vector
        bow = [[count[word] for word in vocabulary] for count in counts]
        return bow
