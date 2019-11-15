import json
import itertools
from nltk import bigrams
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))

    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)

    # return the matrix and the index
    return co_occurrence_matrix, vocab_index


def sort_matrix_by_nb_coocurence(matrix, vocab_index):
    matrix = np.squeeze(np.asarray(matrix))
    sorted_indices_row = np.argsort(matrix.sum(axis=1))[::-1]  # sort indices by row (sum of the elemnts) & reverse list (to be descending order)
    sorted_matrix = matrix[:, sorted_indices_row]

    sorted_indices_col = np.argsort(sorted_matrix.sum(axis=0))[
                     ::-1]  # sort indices by column (sum of the elemnts) & reverse list (to be descending order)
    sorted_matrix = sorted_matrix[:, sorted_indices_col]

    sorted_indices_row_col = [0] * len(sorted_indices_row)
    for i in range(len(sorted_indices_row)):
        pos = sorted_indices_col[i]
        sorted_indices_row_col[i] = sorted_indices_row[pos]

    return sorted_indices_row_col, sorted_matrix



def test_cooccurence_matrix(text_data):
    text_data = [t.split(",") for t in text_data]

    # Create one list using many lists
    data = list(itertools.chain.from_iterable(text_data))
    matrix, vocab_index = generate_co_occurrence_matrix(data)
    sorted_indices, sorted_matrix = sort_matrix_by_nb_coocurence(matrix, vocab_index)
    sorted_vocab_index = {}

    for k in vocab_index.keys():
        sorted_vocab_index[k] = sorted_indices.index(vocab_index[k])


    sorted_vocab_index_v = [v for (v,_) in sorted(sorted_vocab_index.items(), key=lambda kv: kv[1])]

    data_matrix = pd.DataFrame(matrix, index=vocab_index,
                               columns=vocab_index)

    data_matrix_sorted = pd.DataFrame(sorted_matrix, index=sorted_vocab_index,
                               columns=sorted_vocab_index_v)

    data_matrix = data_matrix.drop(columns=['END'])
    data_matrix = data_matrix.drop(['END'])
    data_matrix = data_matrix[(data_matrix.T != 0).any()]
    data_matrix[data_matrix == 0] = np.nan

    data_matrix_sorted = data_matrix_sorted.drop(columns=['END'])
    data_matrix_sorted = data_matrix_sorted.drop(['END'])
    data_matrix_sorted = data_matrix_sorted[(data_matrix_sorted.T != 0).any()]
    data_matrix_sorted[data_matrix_sorted == 0] = np.nan

    column_values = list(data_matrix_sorted.columns.values)
    # print(column_values[0:5])
    # maxValuesObj = data_matrix_sorted.max()
    # print(maxValuesObj)
    first_data_matrix = data_matrix_sorted.nlargest(30, column_values[0]) # sort the rows
    first_data_matrix = first_data_matrix.iloc[:,:30] # first 50 columns

    # first_data_matrix = data_matrix_sorted.iloc[:, :50]  # first 50 columns
    # first_data_matrix = first_data_matrix.iloc[:50, :]  # first 50 rows

    cmap = sns.cm.rocket_r

    # fig, axs = plt.subplots(ncols=2)
    # # sns.heatmap(data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
    # #             linewidths=1, ax = axs[0])
    #
    # sns.heatmap(data_matrix_sorted, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
    #             linewidths=1, ax = axs[0])
    #
    # sns.heatmap(first_data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
    #             linewidths=1, ax=axs[1])

    sns.heatmap(first_data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
                linewidths=1, )

    #sns.clustermap(data_matrix, xticklabels=1, yticklabels=1,cmap=cmap)
    plt.show()

def main():
    with open("data/dict_actions_cooccurence.json") as f:
        dict_actions_cooccurence = json.loads(f.read())

    verbs = dict_actions_cooccurence["verb"]
    verb_particle = dict_actions_cooccurence["verb_particle"]
    verb_particle_nouns = dict_actions_cooccurence["verb_particle_nouns"]
    entire_action = dict_actions_cooccurence["entire_action"]

    test_cooccurence_matrix(verbs)

if __name__ == '__main__':
    main()