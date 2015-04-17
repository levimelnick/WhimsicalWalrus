import sys
from pyspark import SparkContext, SparkConf
import numpy as np


def run(V, W, H, m, n, num_factors, num_workers, num_iterations, beta_val,
        lambda_val, sc, outputW_filepath, outputH_filepath):

    def update(partition):

        tuples = partition.next()

        V_p = tuples[1][0]
        W_p = tuples_to_row_dict(tuples[1][1], num_factors)
        H_p = tuples_to_col_dict(tuples[1][2], num_factors)

        for line in V_p:

            i, j, V_ij = line[0], line[1], line[2]

            W_p_i = W_p[i]
            H_p_j = H_p[j]

            scalar = 2 * (V_ij - np.inner(W_p_i, H_p_j))

            N_i = N_i_dict[i]
            N_j = N_j_dict[j]

            W_p[i] += epsilon * (scalar * H_p_j - 2 * (lambda_val / N_i) * W_p_i)
            H_p[j] += epsilon * (scalar * W_p_i - 2 * (lambda_val / N_j) * H_p_j)

        return (0, row_dict_to_tuples(W_p)), (1, col_dict_to_tuples(H_p))

    iter = 1
    N_i_dict, N_j_dict = get_Ns(V)
    key_V = V.keyBy(lambda i: i[0] % num_workers)
    key_W = W.keyBy(lambda i: i[0] % num_workers)
    key_H = H.keyBy(lambda j: j[1] % num_workers)

    key_V_list = []
    for s in range(num_workers):
        key_V_list.append(key_V.filter(lambda j: (j[1][1] % num_workers) == ((j[1][0] + s) % num_workers)))

    while iter <= num_iterations:
        epsilon = (100 + iter) ** (-beta_val)  # Learning rate

        for s in range(num_workers):
            new_WH = key_V_list[s].groupWith(key_W, key_H).partitionBy(num_workers).mapPartitions(update)

            key_W = new_WH.filter(lambda z: z[0] == 0).map(lambda z: z[1])\
                .flatMap(lambda z: z).keyBy(lambda i: i[0] % num_workers)
            key_H = new_WH.filter(lambda z: z[0] == 1).map(lambda z: z[1])\
                .flatMap(lambda z: z).keyBy(lambda j: ((j[1] - s) % num_workers + num_workers) % num_workers)

        iter += 1

    W = key_W.map(lambda z: z[1])
    H = key_H.map(lambda z: z[1])

    output_results(W, (m, num_factors), outputW_filepath)
    output_results(H, (num_factors, n), outputH_filepath)
    return


def output_results(rdd, shape, output_path):
    res = np.zeros(shape)
    for tup in rdd.collect():
        res[tup[0], tup[1]] = tup[2]

    np.savetxt(output_path, res, delimiter=',')


def get_Ns(rdd):

    N_i = rdd.map(lambda z: (z[0], z[1])).countByKey()
    N_j = rdd.map(lambda z: (z[1], z[0])).countByKey()

    return N_i, N_j


def tuples_to_row_dict(tuples, num_factors):
    d = {}
    for tup in tuples:
        i, j = tup[0], tup[1]
        if i not in d:
            d[i] = np.zeros(num_factors)
        d[i][j] = tup[2]

    return d


def tuples_to_col_dict(tuples, num_factors):
    d = {}
    for tup in tuples:
        i, j = tup[0], tup[1]
        if j not in d:
            d[j] = np.zeros(num_factors)
        d[j][i] = tup[2]

    return d


def row_dict_to_tuples(row_dict):
    tuples = []

    for item in row_dict.items():
        i = item[0]
        col_vals = item[1]
        for j, rating in enumerate(col_vals):
            tuples.append((i, j, rating))

    return tuples


def col_dict_to_tuples(col_dict):
    tuples = []

    for item in col_dict.items():
        j = item[0]
        row_vals = item[1]
        for i, rating in enumerate(row_vals):
            tuples.append((i, j, rating))

    return tuples


def parse_matrix(path, sc):
    # Handle case where path leads to file
    if "." in path.split("/")[-1]:
        return parse_matrix_file(path, sc)
    # Handle case where path is a directory
    return parse_matrix_directory(path, sc)


def parse_matrix_file(path, sc):
    """

    :rtype : numpy array
    :param path: 
    :param sc:
    :return: Numpy array of input matrix V, row dimension, column dimension
    """
    rdd = sc.textFile(path).map(lambda x: map(int, x.split(",")))
    return map_ids(rdd)


def parse_matrix_directory(path, sc):

    """
    Parses all the movie ratings files in directory 'path' to construct the
    input matrix V with user_id as the rows and movie_id as the columns.

    :rtype : (array, int, int)
    :param path: Path to directory with movie files.
    :param sc: The spark context.
    :return: RDD of input matrix V, row dimension, column dimension
    """

    def parse_tuple(tup):
        triples = []
        movie_id, lines = tup[1].split(':\n')
        for line in lines.split('\n'):
            tokens = line.split(',')
            triples.append((int(tokens[0]), int(movie_id), float(tokens[1])))
        return triples

    # Get RDD of list of triples
    rdd = sc.wholeTextFiles(path).flatMap(parse_tuple)

    return map_ids(rdd)

def map_ids(rdd):
    i = 0
    imap = {}
    for user_id in rdd.map(lambda z: z[0]).distinct().sortBy(lambda z: z).collect():
        imap[user_id] = i
        i += 1

    j = 0
    jmap = {}
    for movie_id in rdd.map(lambda z: z[1]).distinct().sortBy(lambda z: z).collect():
        jmap[movie_id] = j
        j += 1

    V = rdd.map(lambda z: (imap[z[0]], jmap[z[1]], z[2]))

    return V, i, j


def initialize_factor_matrix(m, n, num_workers, sc):
    res = []
    for i in range(m):
        for j in range(n):
            res.append((i, j, np.random.random()))
    return sc.parallelize(res, num_workers)


def print_rdd(rdd):
    for x in rdd.collect():
        print(x)


def main():
    # Initialize pyspark context #
    # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    conf = SparkConf().setAppName("DSGD").setMaster("local[4]")
    sc = SparkContext(conf=conf)

    # Initialize variables #
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    # Parse input and initialize factor matrices #
    V, m, n = parse_matrix(inputV_filepath, sc)
    W = initialize_factor_matrix(m, num_factors, num_workers, sc)
    H = initialize_factor_matrix(num_factors, n, num_workers, sc)

    # Run main loop of algorithm
    run(V, W, H, m, n, num_factors, num_workers, num_iterations, beta_value,
        lambda_value, sc, outputW_filepath, outputH_filepath)


if __name__ == "__main__":
    main()

