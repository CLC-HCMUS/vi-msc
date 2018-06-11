import sys
import pulp
import kenlm
import regex
import requests
import collections
import networkx as nx
from wordgraph import WordGraph
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def debug(*args, **kwargs):
    pass


def load_stopwords(file_name):
    stopwords = []
    with open(file_name, 'rb') as f:
        try:
            for line in f:
                line = line.decode('UTF-8').rstrip('\r\n')
                if len(line) > 0 and not line.startswith('#'):
                    stopwords.append(line)
        except Exception:
            pass
    return stopwords


stopwords = load_stopwords('stopwords-vi.txt')
kenlm_model = None


def normalize_special_chars(s):
    s = regex.sub(r"“|”|``|''", '"', s)
    s = regex.sub(r'`|‘|’', "'", s)
    s = regex.sub(r'–', '-', s)
    s = regex.sub(r'…', '...', s)
    s = regex.sub(r'\s+', ' ', s)
    return s.strip()


def remove_underscore(s):
    s = regex.sub(r'([^ ])_([^ ])', '\g<1> \g<2>', s)
    s = regex.sub(r'_([^ ])', '\g<1>', s)
    s = regex.sub(r'([^ ])_', '\g<1>', s)
    s = regex.sub(r'\s+', ' ', s)
    return s.strip()


def normalize_punctuation(s):
    s = regex.sub(r'\s*"\s+([^"]+)\s+"\s*', ' "\g<1>" ', s)
    s = regex.sub(r'\s+([\)\]}])\s*', '\g<1> ', s)
    s = regex.sub(r'\s*([\(\[{])\s+', ' \g<1>', s)
    s = regex.sub(r'\s+([!,.:;?])\s*', '\g<1> ', s)
    s = regex.sub(r'([!,.:;?])\s*(?=[!,.:;?\)\]}"])', '\g<1>', s)
    s = regex.sub(r'\s+', ' ', s)
    return s.strip()


def read_file(file_name):
    with open(file_name, 'rb') as f:
        try:
            return f.read().decode('UTF-8')
        except Exception:
            pass


def write_file(data, file_name):
    with open(file_name, 'wb') as f:
        try:
            f.write(data.encode('UTF-8'))
        except Exception:
            pass


def parse(s):
    try:
        response = requests.post(
            url='http://112.213.86.221:9000/handle', data={'text': s.encode('UTF-8')})
        response.raise_for_status()
        response = response.json()
        if response['status']:
            return response['sentences']
    except Exception:
        pass
    return []


def parse_docs(raw_docs):
    docs = []
    for doc_pos, raw_doc in enumerate(raw_docs):
        sentences = []
        raw_doc = normalize_special_chars(raw_doc)
        if len(raw_doc) == 0:
            continue
        for sentence_pos, sentence in enumerate(parse(raw_doc)):
            tokens = [w['form'] for w in sentence]
            sentences.append({
                'name': doc_pos,
                'pos': sentence_pos,
                'sentence': ' '.join(tokens),
                'tokens': tokens,
                'tags': [w['posTag'] for w in sentence]
            })
        docs.append(sentences)
    return docs


def sentence_similarity(docs):
    vectorizer = TfidfVectorizer(min_df=0, stop_words=stopwords)
    matrix = vectorizer.fit_transform(docs)
    return cosine_similarity(matrix, matrix)


def kenlm_score(sentence):
    return kenlm_model.score(sentence) / (1. + len(sentence.split(' ')))


def eval_linguistic_score(cluster):
    for sentence in cluster:
        sentence['linguistic_score'] = 1. / \
            (1. - kenlm_score(sentence['sentence']))
    return cluster


def eval_linguistic_score_clusters(clusters):
    for cluster in clusters:
        eval_linguistic_score(cluster)
    return clusters


def eval_informativeness_score(cluster):
    sentences = []
    for sentence in cluster:
        sentences.append(sentence['sentence'])

    if len(sentences) > 0:
        # Undirected graph
        graph = nx.from_numpy_matrix(sentence_similarity(sentences))
        informativeness_score = nx.pagerank(graph, max_iter=1000)
        for i, sentence in enumerate(cluster):
            sentence['informativeness_score'] = informativeness_score[i]

    return cluster


def eval_informativeness_score_clusters(clusters):
    for cluster in clusters:
        eval_informativeness_score(cluster)
    return clusters


def clustering_sentences(docs, cluster_threshold=0.5, sim_threshold=0.5, n_top_sentences=None, n_top_clusters=None,
                         use_kmeans=False, num_clusters=8):
    # Number of documents in each cluster
    num_docs = len(docs)
    cluster_threshold *= num_docs

    # Determine important document
    raw_docs = [' '.join([' '.join(s['tokens']) for s in doc]) for doc in docs]
    raw_docs.append(' '.join(raw_docs))

    # Compute cosine similarities and get index of important document
    imp_doc_idx = sentence_similarity(raw_docs)[num_docs:, :num_docs].argmax()

    # Generate clusters
    clusters = []
    raw_sentences = []

    if use_kmeans:
        kmeans_sentences = [' '.join(s['tokens']) for s in docs[imp_doc_idx]]

        # Clustering sentences in important document
        vectorizer = TfidfVectorizer(min_df=0, stop_words=stopwords)
        matrix = vectorizer.fit_transform(kmeans_sentences)
        kmeans = KMeans(n_clusters=min(num_clusters, len(kmeans_sentences)))
        kmeans.fit(matrix)

        imp_cluster_indices = []
        cosine_similarities = cosine_similarity(
            kmeans.cluster_centers_, matrix)
        for row in cosine_similarities:
            imp_cluster_indices.append(row.argmax())

        kmeans_clusters = collections.defaultdict(list)

        # Cluster id
        for i, label in enumerate(kmeans.labels_):
            kmeans_clusters[label].append(i)

        for v in kmeans_clusters.values():
            sentences = []
            for idx in set(v) & set(imp_cluster_indices):
                raw_sentences.append(
                    ' '.join(docs[imp_doc_idx][idx]['tokens']))
                break

            for idx in v:
                sentence = docs[imp_doc_idx][idx]
                sentence['sim'] = 1.0
                sentences.append(sentence)
            clusters.append(sentences)
    else:
        for sentence in docs[imp_doc_idx]:
            sentence['sim'] = 1.0
            clusters.append([sentence])
            raw_sentences.append(' '.join(sentence['tokens']))

    num_clusters = len(clusters)
    debug('- Number of clusters: %d' % num_clusters)  # OK

    # Align sentences in other documents into clusters
    sentence_mapping = []
    for i, doc in enumerate(docs):
        if i == imp_doc_idx:  # Skip important document
            continue

        for sentence in doc:
            sentence_mapping.append(sentence)
            raw_sentences.append(' '.join(sentence['tokens']))

    # Compute cosine similarities and get index of cluster
    cosine_similarities = sentence_similarity(
        raw_sentences)[num_clusters:, :num_clusters]
    for i, row in enumerate(cosine_similarities):
        max_sim = row.max()
        if max_sim >= sim_threshold:
            sentence = sentence_mapping[i]
            sentence['sim'] = max_sim
            clusters[row.argmax()].append(sentence)

    ordered_clusters = []
    if n_top_sentences is None:
        ordered_clusters = clusters
    else:
        for cluster in clusters:
            ordered_cluster = sorted(
                cluster, key=lambda s: s['sim'], reverse=True)
            ordered_clusters.append(ordered_cluster[:n_top_sentences])

    final_clusters = []
    for cluster in ordered_clusters:
        if len(cluster) >= int(cluster_threshold):
            final_clusters.append(cluster)

    if n_top_clusters:
        # Sort clusters by number of sentences in descending order
        cluster_indices = sorted([(i, len(cluster)) for i, cluster in enumerate(final_clusters)], key=lambda k: k[1],
                                 reverse=True)
        cluster_indices = dict(cluster_indices[:n_top_clusters])
        final_clusters = [cluster for i, cluster in enumerate(
            final_clusters) if i in cluster_indices]

    debug('-- Number of clusters after filtering: %d' %
          len(final_clusters))  # OK
    for i, cluster in enumerate(final_clusters):
        debug('---- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    return final_clusters


def remove_similar_sentences(compressed_cluster, original_cluster, sim_threshold=0.8):
    sentences = []
    for sentence in original_cluster:
        sentences.append(sentence['sentence'])
    for sentence in compressed_cluster:
        sentences.append(sentence['sentence'])

    final_cluster = []

    if len(sentences) > 0:
        num_original_sentences = len(original_cluster)

        cosine_similarities = sentence_similarity(
            sentences)[num_original_sentences:, :num_original_sentences]

        for i, row in enumerate(cosine_similarities):
            if row.max() < sim_threshold:
                final_cluster.append(compressed_cluster[i])

    return final_cluster


def remove_similar_sentences_clusters(compressed_clusters, original_clusters, sim_threshold=0.8):
    final_clusters = []
    for i, original_cluster in enumerate(original_clusters):
        cluster = remove_similar_sentences(
            compressed_clusters[i], original_cluster, sim_threshold)
        if len(cluster) > 0:
            final_clusters.append(cluster)

    return final_clusters


def ordering_clusters(clusters):
    edges = []
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            if i == j:
                continue

            cij = 0
            cji = 0
            for sentence_i in cluster_i:
                for sentence_j in cluster_j:
                    if sentence_i['name'] == sentence_j['name']:  # In the same document
                        # Cluster i precedes cluster j
                        if sentence_i['pos'] < sentence_j['pos']:
                            cij += 1
                        # Cluster j precedes cluster i
                        elif sentence_i['pos'] > sentence_j['pos']:
                            cji += 1

            edges.append((i, j, cij))
            edges.append((j, i, cji))

    # Return if empty
    if len(edges) == 0:
        return clusters

    # Find optimal path
    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(edges)

    def compute_node_weights(graph):
        values = {}
        for v in graph:
            values[v] = graph.out_degree(
                v, 'weight') - graph.in_degree(v, 'weight')
        return values

    cluster_order = []
    while digraph.number_of_nodes() > 0:
        nodes = compute_node_weights(graph=digraph)
        # Using the original order if there is more max values
        node = max(nodes, key=nodes.get)
        cluster_order.append((node, nodes[node]))
        digraph.remove_node(node)

    debug('-- Order:', cluster_order)

    final_clusters = []
    for i, _ in cluster_order:
        final_clusters.append(clusters[i])

    return final_clusters


def compress_clusters(clusters, num_words=8, num_candidates=200, sim_threshold=0.8):
    compressed_clusters = []

    for cluster in clusters:
        sentences = []
        for sentence in cluster:
            sentences.append(
                ' '.join(['%s/%s' % (token, sentence['tags'][i]) for i, token in enumerate(sentence['tokens'])]))

        compresser = WordGraph(
            sentence_list=sentences, stopwords=stopwords, nb_words=num_words)
        candidates = compresser.get_compression(nb_candidates=num_candidates)

        compressed_cluster = []
        for score, candidate in candidates:
            tokens = [w[0] for w in candidate]
            sentence = ' '.join(tokens)

            compressed_cluster.append({
                'num_words': len(tokens),
                'score': score / len(tokens),
                'sentence': sentence
            })

        if len(compressed_cluster) > 0:
            compressed_clusters.append(compressed_cluster)

    debug('-- Number of sentences in each cluster:')
    for i, cluster in enumerate(compressed_clusters):
        debug('---- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    compressed_clusters = remove_similar_sentences_clusters(
        compressed_clusters, clusters, sim_threshold=sim_threshold)

    debug('-- Number of sentences in each cluster after removing similar sentences:')
    for i, cluster in enumerate(compressed_clusters):
        debug('---- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    return compressed_clusters


def reduce_cluster_size(cluster, keep_num=3):
    # Sort cluster by score
    cluster = sorted(
        cluster, key=lambda s: s['informativeness_score'] * s['linguistic_score'], reverse=True)

    size_mapping = {}
    final_cluster = []

    for sentence in cluster:
        num_words = sentence['num_words']
        # Keep keep_num sentence with the same size
        if size_mapping.get(num_words, 0) < keep_num:
            final_cluster.append(sentence)
            size_mapping[num_words] = size_mapping.get(num_words, 0) + 1

    return final_cluster


def solve_ilp(clusters, num_words=100, sim_threshold=0.5, reduce_clusters_size=False, keep_num=3):
    # Reduce clusters size
    if reduce_clusters_size:
        clusters = [reduce_cluster_size(cluster, keep_num)
                    for cluster in clusters]

        debug('-- Number of sentences in each cluster after reducing cluster size:')
        for i, cluster in enumerate(clusters):
            debug('---- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    # Define problem
    ilp_problem = pulp.LpProblem("ILPSumNLP", pulp.LpMaximize)

    # For storing
    sentences = []
    ilp_vars_matrix = []

    # For creating constraint
    obj_function = []
    length_constraint = []

    # Iterate over the clusters
    for i, cluster in enumerate(clusters):

        ilp_vars = []

        # Iterate over the sentence in the clusters
        for j, sentence in enumerate(cluster):
            var = pulp.LpVariable('var_%d_%d' % (i, j), cat=pulp.LpBinary)

            # Prepare objective function
            obj_function.append(
                sentence['informativeness_score'] * sentence['linguistic_score'] * var)

            # Prepare constraints
            length_constraint.append(sentence['num_words'] * var)

            # Store ILP variable
            ilp_vars.append(var)

            # Store sentence
            sentences.append(sentence['sentence'])

        # Store ILP variables
        ilp_vars_matrix.append(ilp_vars)

        # Create constraint
        ilp_problem += pulp.lpSum(ilp_vars) <= 1.0, 'Cluster_%d constraint' % i

    # Create constraint
    ilp_problem += pulp.lpSum(length_constraint) <= num_words, 'Length constraint'

    # Create objective function
    ilp_problem += pulp.lpSum(obj_function), 'Objective function'

    # Compute cosine similarities
    cosine_similarities = sentence_similarity(sentences)

    # Filter similar sentence between clusters
    pos = 0
    for i, cluster in enumerate(clusters):
        _pos = 0
        for _i, _cluster in enumerate(clusters):
            if i != _i:
                for j, sentence in enumerate(cluster):
                    for _j, _sentence in enumerate(_cluster):
                        if cosine_similarities[pos + j][_pos + _j] >= sim_threshold:
                            ilp_problem += ilp_vars_matrix[i][j] + ilp_vars_matrix[_i][_j] <= 1.0, \
                                'Sim(var_%d_%d,var_%d_%d) constraint' % (
                                    i, j, _i, _j)

            _pos += len(_cluster)
        pos += len(cluster)

    # Maximizing objective function
    ilp_problem.solve(pulp.GLPK(msg=0))

    # Create summary
    final_sentences = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            if ilp_vars_matrix[i][j].varValue == 1.0:
                final_sentences.append(clusters[i][j]['sentence'])

    return final_sentences


def main():
    # Args
    mode, inp, oup = sys.argv[1], sys.argv[2], sys.argv[3]

    # Result
    compressed_sentences = []

    # Process
    try:
        # Modes
        if mode == 'msc':
            docs = parse_docs([read_file(inp)])
            if len(docs) == 0:
                return

            sentences = [' '.join([w + '/' + s['tags'][i]
                                   for i, w in enumerate(s['tokens'])]) for s in docs[0]]

            compresser = WordGraph(
                sentence_list=sentences, stopwords=stopwords)
            candidates = compresser.get_compression(nb_candidates=250)

            sentences = []
            for score, candidate in candidates:
                sentences.append({
                    'score': score / len(candidate),
                    'sentence': ' '.join([w[0] for w in candidate])
                })
            sorted_sentences = sorted(
                sentences, key=lambda k: k['score'])

            for s in sorted_sentences[:1]:
                compressed_sentences.append(normalize_punctuation(
                    remove_underscore(s['sentence'])))
        else:
            docs = regex.split(r'\s*={5}\s*', read_file(inp))
            docs = parse_docs(docs)
            if len(docs) == 0:
                return

            clusters = clustering_sentences(docs=docs, cluster_threshold=0.7, sim_threshold=0.18,
                                            n_top_sentences=None, n_top_clusters=None, use_kmeans=True, num_clusters=12)
            clusters = ordering_clusters(clusters=clusters)
            compressed_clusters = compress_clusters(
                clusters=clusters, num_words=8, num_candidates=250, sim_threshold=0.8)

            global kenlm_model
            kenlm_model = kenlm.Model('vi.bin')

            scored_clusters = eval_linguistic_score_clusters(
                compressed_clusters)
            scored_clusters = eval_informativeness_score_clusters(
                scored_clusters)

            final_sentences = solve_ilp(clusters=scored_clusters, num_words=120, sim_threshold=0.5,
                                        reduce_clusters_size=False)

            for final_sentence in final_sentences:
                final_sentence = normalize_punctuation(
                    remove_underscore(final_sentence))
                compressed_sentences.append(final_sentence)

    except Exception:
        compressed_sentences = []
    finally:
        write_file(' '.join(compressed_sentences), oup)


if __name__ == '__main__':
    main()
