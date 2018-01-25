from pyspark import SparkConf, SparkContext
import sys
import itertools


def list_common_friends(file_input, file_output):

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(file_input)

    # Gives us the exhaustive "potential friends in common" pairs,
    # without beeing sure that the element of the tuple are direct friends.

    undirect_friends = lines.flatMap(lambda g: [(s, 1) for s in itertools.combinations(g.split("\t")[1].split(","), 2)])
    undirect_friends = undirect_friends.reduceByKey(lambda n1, n2: n1 + n2)

    # Gives us the list of direct friends  pairs
    direct_friends = lines.flatMap(lambda g: [((g.split("\t")[0], s), 1) for s in g.split("\t")[1].split(",")])

    # We substract the possible pairs which are direct friends
    undirect_friends = undirect_friends.subtractByKey(direct_friends)

    # We change the maping such as we get for each user (the key) a candidate with the number of common friends
    # (A,B) and (B,A) to be symetric (combinations above doesnt give the symetrics)
    undirect_friends = undirect_friends.flatMap(lambda g: [(g[0][0], (g[0][1], g[1])), (g[0][1], (g[0][0], g[1]))])

    # We group by user the list of candidates, sort by decreasing number of common friends and take the first 10
    undirect_friends = undirect_friends.groupByKey().map(
        lambda x: (x[0], [p[0] for p in sorted(list(x[1]), key=lambda y: (-float(y[1]), float(y[0])))][:10]))

    text_file = open(file_output, "w")

    for pair in undirect_friends.collect():
        text_file.write("%s    %s\n" % ("'"+pair[0]+"'", ",".join([n for n in pair[1]])))

    sc.stop()


# test function
# def check_common_friends(user1, user2):
#     f = open("/Users/maximedumonal/Github/Stanford/CS246/data/soc-LiveJournal1Adj.txt", 'rU')
#     g = open("/Users/maximedumonal/Github/Stanford/CS246/data/soc-LiveJournal1Adj.txt", 'rU')
#     for line in f:
#         linesplit = line.strip().split("\t")
#         if linesplit[0] == user1:
#             output_1 = linesplit[1].split(",")
#             break
#
#     for line in g:
#         linesplit = line.strip().split("\t")
#         if linesplit[0] == user2:
#             output_2 = linesplit[1].split(",")
#             break
#
#     return set(output_1).intersection(output_2) #, set(output_1), set(output_2)


if __name__ == '__main__':
    list_common_friends(sys.argv[1], sys.argv[2])
