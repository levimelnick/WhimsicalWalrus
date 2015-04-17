import sys
from pyspark import SparkContext, SparkConf
import numpy as np
from functools import partial
from scipy import sparse

_ITERATION_LOSS = False
_TOTAL_LOSS = False


def run(v, w, h, m, n, num_factors, num_workers, num_iterations, beta_val,
        lambda_val, sc, output_w_filepath, output_h_filepath):
    """
    Main loop of DSDG-MF algorithm. Creates strata from data, partitions them
    to workers, performs SGD updates on blocks, writes results to file.

    :param v: RDD of V matrix.
    :param w: RDD of W matrix.
    :param h: RDD of H matrix.
    :param m: Row dimension of V matrix.
    :param n: Column dimension of V matrix.
    :param num_factors: Number of factors.
    :param num_workers: Number of Workers.
    :param num_iterations: Number of iterations.
    :param beta_val: Beta.
    :param lambda_val: Lambda.
    :param sc: Spark context.
    :param output_w_filepath: Output filepath for W.
    :param output_h_filepath: Output filepath of H.
    :return: None.
    """

    iteration = 0
    loss_output_str = ""

    # Broadcast "global" variables
    n_i_dict, n_j_dict = get_ns(v)
    n_i_dict_bc = sc.broadcast(n_i_dict)
    n_j_dict_bc = sc.broadcast(n_j_dict)
    lambda_val_bc = sc.broadcast(lambda_val)
    num_factors_bc = sc.broadcast(num_factors)
    beta_val_bc = sc.broadcast(beta_val)

    # Set initial key values for v, w, and h
    key_v = v.keyBy(lambda i: i[0] % num_workers)
    key_w = w.keyBy(lambda i: i[0] % num_workers)
    key_h = h.keyBy(lambda j: j[1] % num_workers)

    # Pre-filter V and store result in list of RDDs
    key_v_list = []
    for s in range(num_workers):
        key_v_list.append(key_v.filter(lambda j: (j[1][1] % num_workers) == ((j[1][0] + s) % num_workers)))

    while iteration < num_iterations:
        iteration_bc = sc.broadcast(iteration)

        for s in range(num_workers):

            new_wh = key_v_list[s].groupWith(key_w, key_h).partitionBy(num_workers)\
                .mapPartitions(partial(update
                                       , iteration=iteration_bc
                                       , n_i=n_i_dict_bc
                                       , n_j=n_j_dict_bc
                                       , lambda_val=lambda_val_bc
                                       , num_factors=num_factors_bc
                                       , beta_val=beta_val_bc))

            key_w = new_wh.filter(lambda z: z[0] == 0).map(lambda z: z[1])\
                .flatMap(lambda z: z).keyBy(lambda i: i[0] % num_workers)
            key_h = new_wh.filter(lambda z: z[0] == 1).map(lambda z: z[1])\
                .flatMap(lambda z: z).keyBy(lambda j: (j[1] - s) % num_workers)

        if _ITERATION_LOSS:
            # Calculate per iteration loss
            iteration_loss = get_loss(key_v, key_w, key_h)
            loss_output_str += str(iteration) + "," + str(iteration_loss) + "\n"

        iteration += 1

    if _TOTAL_LOSS:
        total_loss = get_loss(key_v, key_w, key_h)

    w = key_w.map(lambda z: z[1])
    h = key_h.map(lambda z: z[1])

    # Output w and h matrices to csv
    output_results(w, (m, num_factors), output_w_filepath)
    output_results(h, (num_factors, n), output_h_filepath)

    # Write record of loss to file
    with open(str(num_factors) + "factors.txt", "w") as loss_file:
        if _ITERATION_LOSS:
            loss_file.write(loss_output_str)
        if _TOTAL_LOSS:
            loss_file.write(str(total_loss))
    return


def update(partition, iteration, n_i, n_j, lambda_val, num_factors, beta_val):
    """

    :param partition: Iterator of one tuple representing key and block.
    :param iteration: Broadcast object of iteration number.
    :param n_i: Broadcast object of N_i dictionary.
    :param n_j: Broadcast object of N_j dictionary.
    :param lambda_val: Broadcast object of lambda value.
    :param num_factors: Broadcast object of number of factors.
    :param beta_val: Broadcast object of beta value.
    :return: tuple of (key, list of tuples) for new w and h
    """
    tuples = partition.next()

    iteration = iteration.value
    n_i = n_i.value
    n_j = n_j.value
    lambda_val = lambda_val.value
    num_factors = num_factors.value
    beta_val = beta_val.value

    num_updates = iteration * sum(n_i.values())

    v_p = tuples[1][0]
    w_p = tuples_to_row_dict(tuples[1][1], num_factors)
    h_p = tuples_to_col_dict(tuples[1][2], num_factors)

    step = 0

    for line in v_p:
        epsilon = (1000 + num_updates + step) ** (-beta_val)  # Learning rate

        i, j, v_ij = line[0], line[1], line[2]

        w_p_i = w_p[i]
        h_p_j = h_p[j]

        unsquared_loss = v_ij - np.inner(w_p_i, h_p_j)

        n_i_val = n_i[i]
        n_j_val = n_j[j]

        w_p[i] += epsilon * (2 * unsquared_loss * h_p_j - 2 * (lambda_val / n_i_val) * w_p_i)
        h_p[j] += epsilon * (2 * unsquared_loss * w_p_i - 2 * (lambda_val / n_j_val) * h_p_j)

        step += 1

    return (0, row_dict_to_tuples(w_p)), (1, col_dict_to_tuples(h_p))


def output_results(rdd, shape, output_path):
    """
    Writes RDD matrix to csv.

    :param rdd: RDD of matrix.
    :param shape: Tuple of dimensions of matrix.
    :param output_path: filename
    :return:
    """

    res = np.zeros(shape)
    for tup in rdd.collect():
        res[tup[0], tup[1]] = tup[2]

    np.savetxt(output_path, res, delimiter=',')


def get_loss(key_v, key_w, key_h):
    """
    Calculate the loss from RDDs representing v, w, and h. Adapted
    from eval_acc.py.

    :param key_v: RDD representing v with keys.
    :param key_w: RDD representing w with keys.
    :param key_h: RDD representing h with keys.
    :return: loss (float)
    """

    v, select = load_sparse_matrix(key_v)
    w, notused1 = load_sparse_matrix(key_w)
    h, notused2 = load_sparse_matrix(key_h)

    diff = v - w * h
    loss = 0
    for i, j in select:
        loss += diff[i, j] * diff[i, j]
    return loss


def load_sparse_matrix(key_rdd):
    """
    Convert RDD to csr_matrix. Adapted from eval_acc.py

    :param key_rdd: RDD with keys representing a matrix.
    :return: csr_matrix
    """

    val = []
    row = []
    col = []
    select = []
    for tup in key_rdd.map(lambda z: z[1]).collect():
        row.append(tup[0])
        col.append(tup[1])
        val.append(tup[2])
        select.append( (tup[0], tup[1]) )
    return sparse.csr_matrix((val, (row, col))), select


def get_ns(rdd):
    """
    Get dictionary of N_i and N_j values for calculating the
    regularization penalty.

    :param rdd: RDD representing V.
    :return: Dictionaries of N_i and N_j values.
    """

    n_i = rdd.map(lambda z: (z[0], z[1])).countByKey()
    n_j = rdd.map(lambda z: (z[1], z[0])).countByKey()

    return n_i, n_j


def tuples_to_row_dict(tuples, num_factors):
    """
    Express a list of tuples representing the rows of a matrix as
    a dictionary of numpy arrays representing the rows of a matrix.

    :param tuples: List of tuples
    :param num_factors: Number of factors
    :return: Dictionary of numpy arrays
    """

    d = {}
    for tup in tuples:
        i, j = tup[0], tup[1]
        if i not in d:
            d[i] = np.zeros(num_factors)
        d[i][j] = tup[2]

    return d


def tuples_to_col_dict(tuples, num_factors):
    """
    Express a list of tuples representing the columns of a matrix as
    a dictionary of numpy arrays representing the columns of a matrix.

    :param tuples: List of tuples
    :param num_factors: Number of factors
    :return: Dictionary of numpy arrays
    """

    d = {}
    for tup in tuples:
        i, j = tup[0], tup[1]
        if j not in d:
            d[j] = np.zeros(num_factors)
        d[j][i] = tup[2]

    return d


def row_dict_to_tuples(row_dict):
    """
    Express a dictionary of numpy arrays representing the rows of a matrix
    as a list of tuples representing the matrix.

    :param col_dict: Dictionary of numpy arrays
    :return: List of tuples
    """

    tuples = []

    for item in row_dict.items():
        i = item[0]
        col_values = item[1]
        for j, rating in enumerate(col_values):
            tuples.append((i, j, rating))

    return tuples


def col_dict_to_tuples(col_dict):
    """
    Express a dictionary of numpy arrays representing the columns of a matrix
    as a list of tuples representing the matrix.

    :param col_dict: Dictionary of numpy arrays
    :return: List of tuples
    """

    tuples = []

    for item in col_dict.items():
        j = item[0]
        row_values = item[1]
        for i, rating in enumerate(row_values):
            tuples.append((i, j, rating))

    return tuples


def parse_matrix(path, sc):
    """
    Parse matrix from file or directory. See parse_matrix_file and
    parse_matrix_directory.

    :param path:
    :param sc:
    :return:
    """

    # Handle case where path leads to file
    if "." in path.split("/")[-1]:
        return parse_matrix_file(path, sc)
    # Handle case where path is a directory
    return parse_matrix_directory(path, sc)


def parse_matrix_file(path, sc):
    """
    Parses a matrix from the file at path.

    :param path: Filepath.
    :param sc: Spark Context.
    :return: RDD representing matrix.
    """

    rdd = sc.textFile(path).map(lambda x: map(int, x.split(",")))
    return map_ids(rdd)


def parse_matrix_directory(path, sc):

    """
    Parses all the movie ratings files in directory 'path' to construct the
    input matrix V with user_id as the rows and movie_id as the columns.

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
    """
    Map movie and user ids from V to sequential ids beginning at 0. Return
    RDD with new mapping and row and column dimensions.

    :param rdd: RDD object representing V.
    :return: (RDD, number of rows, number of columns)
    """

    i = 0
    i_map = {}
    for user_id in rdd.map(lambda z: z[0]).distinct().sortBy(lambda z: z).collect():
        i_map[user_id] = i
        i += 1

    j = 0
    j_map = {}
    for movie_id in rdd.map(lambda z: z[1]).distinct().sortBy(lambda z: z).collect():
        j_map[movie_id] = j
        j += 1

    v = rdd.map(lambda z: (i_map[z[0]], j_map[z[1]], z[2]))

    return v, i, j


def initialize_factor_matrix(m, n, num_workers, sc):
    """
    Initialize a factor matrix of dimension m x n with random values between 0 and 1.

    :param m: Number of rows.
    :param n: Number of columns.
    :param num_workers: Number of workers.
    :param sc: Spark context.
    :return: An RDD object representing the factor matrix.
    """
    res = []
    for i in range(m):
        for j in range(n):
            res.append((i, j, np.random.random()))
    return sc.parallelize(res, num_workers)


def print_rdd(rdd):
    """
    Print the contents of an rdd. Used for debugging only.

    :param rdd: An RDD object.
    :return: None.
    """

    for x in rdd.collect():
        print(x)


def main():
    # Initialize variables #
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    input_v_filepath = sys.argv[6]
    output_w_filepath = sys.argv[7]
    output_h_filepath = sys.argv[8]

    # Initialize pyspark context #
    conf = SparkConf().setAppName("DSGD").setMaster("local["+str(num_workers)+"]")
    sc = SparkContext(conf=conf)

    # Parse input and initialize factor matrices #
    v, m, n = parse_matrix(input_v_filepath, sc)
    w = initialize_factor_matrix(m, num_factors, num_workers, sc)
    h = initialize_factor_matrix(num_factors, n, num_workers, sc)

    # Run main loop of algorithm
    run(v, w, h, m, n, num_factors, num_workers, num_iterations, beta_value,
        lambda_value, sc, output_w_filepath, output_h_filepath)


if __name__ == "__main__":
    main()

