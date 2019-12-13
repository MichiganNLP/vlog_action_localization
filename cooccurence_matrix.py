import json
import itertools
from nltk import bigrams
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

from compute_text_embeddings import create_elmo_embedding, NumpyEncoder

plt.style.use('ggplot')


def cluster_co_occurrence_matrix(corpus, keywords):
    vocab = list(set(corpus))
    bi_grams = list(bigrams(corpus))
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    word_freq = nltk.FreqDist(corpus).most_common(len(vocab))
    total_word = len(corpus)
    co_occurrence_matrix = np.zeros((len(vocab) - 1, len(vocab) - 1))
    vocab_freq = {word[0]: word[1] for word in word_freq}
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = math.log(bigram[1] / (vocab_freq[current] * vocab_freq[previous]) * total_word)
        count = round(count, 1)
        if not (current == 'END' or previous == 'END'):
            co_occurrence_matrix[len(vocab) - 2 - int(current)][int(previous)] = count
    index = keywords[:: -1]
    column = list(keywords)
    data_matrix = pd.DataFrame(co_occurrence_matrix, index=index,
                               columns=column)
    cmap = sns.cm.rocket_r
    data_matrix[data_matrix == 0] = np.nan
    plt.figure(figsize=(15, 15))
    cmap = "RdBu_r"
    sns.set(font_scale=1.6)
    sns.heatmap(data_matrix, annot=False, center=0, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
                linewidths=1)  # , annot_kws={"size": 13})
    plt.show()


def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = sorted(list(vocab))
    vocab_index = {word: i for i, word in enumerate(vocab)}
    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))

    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    word_freq = nltk.FreqDist(corpus).most_common(len(vocab))
    total_word = len(corpus)
    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
    vocab_freq = {word[0]: word[1] for word in word_freq}
    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        # count = math.log(bigram[1] / (vocab_freq[current] * vocab_freq[previous]) * total_word)
        # count = math.log(bigram[1])
        count = bigram[1]
        count = round(count, 3)
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)

    # return the matrix and the index
    return co_occurrence_matrix, vocab_index


def sort_matrix_by_nb_coocurence(matrix, vocab_index):
    matrix = np.squeeze(np.asarray(matrix))
    sorted_indices_row = np.argsort(matrix.sum(axis=1))[
                         ::-1]  # sort indices by row (sum of the elemnts) & reverse list (to be descending order)
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

    sorted_vocab_index_v = [v for (v, _) in sorted(sorted_vocab_index.items(), key=lambda kv: kv[1])]

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

    # column_values = list(data_matrix_sorted.columns.values)
    # print(column_values[0:5])
    # maxValuesObj = data_matrix_sorted.max()
    # print(maxValuesObj)
    # first_data_matrix = data_matrix_sorted.nlargest(30, column_values[0])  # sort the rows
    # first_data_matrix = first_data_matrix.iloc[:, :30]  # first 50 columns
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
    plt.figure(figsize=(15, 15))
    sns.heatmap(data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
                linewidths=1, )

    # sns.clustermap(data_matrix, xticklabels=1, yticklabels=1,cmap=cmap)
    plt.show()


def entire_action_cluster(entire_action, out_path):
    action_cluster = {}
    with open(out_path) as fp:
        for line in fp.readlines():
            parts = line.strip().split()
            action = ""
            for i in range(len(parts)):
                if i == len(parts) - 1:
                    action_cluster[action[: -1]] = parts[i]
                else:
                    action = action + parts[i] + ' '
    # print(len(action_cluster))
    # print(len(set(entire_action)))
    clusters = []
    for action in entire_action:
        if action in action_cluster:
            clusters.append(action_cluster[action])
        else:
            clusters.append(action)
    return clusters


def cluster_name(cluster_document_name, nb_clusters):
    cluster_action = {}
    with open(cluster_document_name) as fp:
        for line in fp.readlines():
            parts = line.strip().split()
            action = ""
            for i in range(len(parts)):
                if i == len(parts) - 1:
                    if int(parts[-1]) not in cluster_action:
                        cluster_action[int(parts[-1])] = []
                    cluster_action[int(parts[-1])].append(action)
                else:
                    action = action + parts[i] + ' '
    action_documents = []
    for i in range(len(cluster_action)):
        actions = ""
        for j in cluster_action[i]:
            actions = actions + j
        action_documents.append(actions)
    docs = action_documents
    cv = CountVectorizer(stop_words='english')
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    count_vector = cv.transform(docs)
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()
    keywords = []
    for i in range(nb_clusters):
        first_document_vector = tf_idf_vector[i]
        df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
        df = df.sort_values(by=["tfidf"], ascending=False)
        keyword = list(df.head(3).index)
        seperator = ' '
        keywords.append(seperator.join(keyword))
    return keywords


def save_dict_action_cluster(entire_action, clusters, keywords, path_out):
    dict_actions_clusters = {}
    dict_elmo_emmb = {}
    # can also do glove, bert, dnt embeddings
    for keyword in keywords:
        dict_elmo_emmb[keyword] = create_elmo_embedding(keyword)

    for (action, cluster) in zip(entire_action, clusters):
        if cluster != 'END':
            keyword = keywords[int(cluster)]
            cluster_name_emb = dict_elmo_emmb[keyword]
            dict_actions_clusters[action] = [int(cluster), keyword, cluster_name_emb]

    with open(path_out, 'w+') as outfile:
        json.dump(dict_actions_clusters, outfile, cls=NumpyEncoder)


def main():
    with open("/local/oignat/Action_Recog/action_recog_2/steve_human_action/dict_actions_cooccurence.json") as f:
        dict_actions_cooccurence = json.loads(f.read())

    for k in [30, 50, 100, 150, 200, 250, 300]:
        out_path = "/local/oignat/Action_Recog/action_recog_2/steve_human_action/clusters/kmeans_clusters_" + \
                   str(k) + "_DNT.out"

        # test_cooccurence_matrix(dict_actions_cooccurence["verb"])
        # test_cooccurence_matrix(dict_actions_cooccurence["all_actions"])

        # verb_particle = dict_actions_cooccurence["verb_particle"]
        # verb_particle_nouns = dict_actions_cooccurence["verb_particle_nouns"]
        entire_action = dict_actions_cooccurence["all_actions"]

        clusters = entire_action_cluster(entire_action, out_path)
        # test_cooccurence_matrix(clusters)
        keywords = cluster_name(out_path, k)

        save_dict_action_cluster(entire_action, clusters, keywords,
                                 "/local/oignat/Action_Recog/action_recog_2/steve_human_action/cluster_results/dict_actions_clusters_" +
                                 str(k) + ".json")

    # keywords = range(0, 30)
    # cluster_co_occurrence_matrix(clusters, keywords)


if __name__ == '__main__':
    main()

# import json
# import itertools
# from nltk import bigrams
# import numpy as np
# import nltk
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# plt.style.use('ggplot')
#
# def generate_co_occurrence_matrix(corpus):
#     vocab = set(corpus)
#     vocab = list(vocab)
#     vocab_index = {word: i for i, word in enumerate(vocab)}
#
#     # Create bigrams from all words in corpus
#     bi_grams = list(bigrams(corpus))
#
#     # Frequency distribution of bigrams ((word1, word2), num_occurrences)
#     bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
#
#     # Initialise co-occurrence matrix
#     # co_occurrence_matrix[current][previous]
#     co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
#
#     # Loop through the bigrams taking the current and previous word,
#     # and the number of occurrences of the bigram.
#     for bigram in bigram_freq:
#         current = bigram[0][1]
#         previous = bigram[0][0]
#         count = bigram[1]
#         pos_current = vocab_index[current]
#         pos_previous = vocab_index[previous]
#         co_occurrence_matrix[pos_current][pos_previous] = count
#     co_occurrence_matrix = np.matrix(co_occurrence_matrix)
#
#     # return the matrix and the index
#     return co_occurrence_matrix, vocab_index
#
#
# def sort_matrix_by_nb_coocurence(matrix, vocab_index):
#     matrix = np.squeeze(np.asarray(matrix))
#     sorted_indices_row = np.argsort(matrix.sum(axis=1))[::-1]  # sort indices by row (sum of the elemnts) & reverse list (to be descending order)
#     sorted_matrix = matrix[:, sorted_indices_row]
#
#     sorted_indices_col = np.argsort(sorted_matrix.sum(axis=0))[
#                      ::-1]  # sort indices by column (sum of the elemnts) & reverse list (to be descending order)
#     sorted_matrix = sorted_matrix[:, sorted_indices_col]
#
#     sorted_indices_row_col = [0] * len(sorted_indices_row)
#     for i in range(len(sorted_indices_row)):
#         pos = sorted_indices_col[i]
#         sorted_indices_row_col[i] = sorted_indices_row[pos]
#
#     return sorted_indices_row_col, sorted_matrix
#
#
#
# def test_cooccurence_matrix(text_data):
#     text_data = [t.split(",") for t in text_data]
#
#     # Create one list using many lists
#     data = list(itertools.chain.from_iterable(text_data))
#     matrix, vocab_index = generate_co_occurrence_matrix(data)
#     sorted_indices, sorted_matrix = sort_matrix_by_nb_coocurence(matrix, vocab_index)
#     sorted_vocab_index = {}
#
#     for k in vocab_index.keys():
#         sorted_vocab_index[k] = sorted_indices.index(vocab_index[k])
#
#
#     sorted_vocab_index_v = [v for (v,_) in sorted(sorted_vocab_index.items(), key=lambda kv: kv[1])]
#
#     data_matrix = pd.DataFrame(matrix, index=vocab_index,
#                                columns=vocab_index)
#
#     data_matrix_sorted = pd.DataFrame(sorted_matrix, index=sorted_vocab_index,
#                                columns=sorted_vocab_index_v)
#
#     data_matrix = data_matrix.drop(columns=['END'])
#     data_matrix = data_matrix.drop(['END'])
#     data_matrix = data_matrix[(data_matrix.T != 0).any()]
#     data_matrix[data_matrix == 0] = np.nan
#
#     data_matrix_sorted = data_matrix_sorted.drop(columns=['END'])
#     data_matrix_sorted = data_matrix_sorted.drop(['END'])
#     data_matrix_sorted = data_matrix_sorted[(data_matrix_sorted.T != 0).any()]
#     data_matrix_sorted[data_matrix_sorted == 0] = np.nan
#
#     column_values = list(data_matrix_sorted.columns.values)
#     # print(column_values[0:5])
#     # maxValuesObj = data_matrix_sorted.max()
#     # print(maxValuesObj)
#     first_data_matrix = data_matrix_sorted.nlargest(30, column_values[0]) # sort the rows
#     first_data_matrix = first_data_matrix.iloc[:,:30] # first 50 columns
#
#     # first_data_matrix = data_matrix_sorted.iloc[:, :50]  # first 50 columns
#     # first_data_matrix = first_data_matrix.iloc[:50, :]  # first 50 rows
#
#     cmap = sns.cm.rocket_r
#
#     # fig, axs = plt.subplots(ncols=2)
#     # # sns.heatmap(data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
#     # #             linewidths=1, ax = axs[0])
#     #
#     # sns.heatmap(data_matrix_sorted, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
#     #             linewidths=1, ax = axs[0])
#     #
#     # sns.heatmap(first_data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
#     #             linewidths=1, ax=axs[1])
#
#     sns.heatmap(first_data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
#                 linewidths=1, )
#
#     #sns.clustermap(data_matrix, xticklabels=1, yticklabels=1,cmap=cmap)
#     plt.show()
#
# def main():
#     with open("data/dict_actions_cooccurence.json") as f:
#         dict_actions_cooccurence = json.loads(f.read())
#
#     verbs = dict_actions_cooccurence["verb"]
#     verb_particle = dict_actions_cooccurence["verb_particle"]
#     verb_particle_nouns = dict_actions_cooccurence["verb_particle_nouns"]
#     entire_action = dict_actions_cooccurence["entire_action"]
#
#     test_cooccurence_matrix(verbs)
#
# if __name__ == '__main__':
#     main()
