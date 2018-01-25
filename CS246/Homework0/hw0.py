from pyspark import SparkConf, SparkContext
import sys
import re


def count_words(file_input, file_output):

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(file_input)
    print "%d lines" % lines.count()

    words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))
    pairs = words.map(lambda w: (w, 1))
    counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
    counts.saveAsTextFile(file_output)
    sc.stop()


def count_words_by_letter(file_input, file_output):

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(file_input)
    print "%d lines" % lines.count()

    words = lines.flatMap(lambda l: [w.lower() for w in re.split(r'[^\w]+', l)])
    words_first_letter = words.flatMap(lambda w: w[0] if not w == "" and w[0].isalpha() else "")
    pairs = words_first_letter.map(lambda l: (l, 1))
    counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
    counts.saveAsTextFile(file_output)
    sc.stop()


if __name__ == '__main__':
    count_words_by_letter(sys.argv[1], sys.argv[2])
