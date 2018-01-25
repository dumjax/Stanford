import itertools
from collections import defaultdict


def frequent_items(input_file, support):

    f = open(input_file, 'rU')
    baskets = []
    for line in f:
        # we take each element of the basket only once
        baskets.append(list(sorted(set(line.strip().split(" ")))))

    # we count number of occurences per single item
    candidates = defaultdict(lambda: 0)
    for basket in baskets:
        candidates_in_basket = list(set(basket))
        for it in candidates_in_basket:
            candidates[it] += 1

    # we save the frequent ones
    frequent_items_dict = dict()
    for k in candidates:
        if candidates[k] >= support:
            frequent_items_dict[k] = candidates[k]

    # we count the number of occurences of the pairs candidates
    candidates = defaultdict(lambda: 0)
    for basket in baskets:
        candidates_in_basket = sorted(list(set(basket).intersection(set(frequent_items_dict.keys()))))
        candidates_in_basket = list(set([it for it in itertools.combinations(candidates_in_basket, 2)]))
        for it in candidates_in_basket:
            candidates[it] += 1

    # we save the frequent ones
    frequent_pairs_dict = dict()
    for pair in candidates:
        if candidates[pair] >= support:
            frequent_pairs_dict[pair] = candidates[pair]

    # Calculation of confidences
    # (A,B): confidence means A | B or A knowing B
    frequent_pairs_confidence = dict()
    for k in frequent_pairs_dict.keys():
        frequent_pairs_confidence[(k[0], k[1])] = float(frequent_pairs_dict[k]) / frequent_items_dict[k[1]]
        frequent_pairs_confidence[(k[1], k[0])] = float(frequent_pairs_dict[k]) / frequent_items_dict[k[0]]

    for pair_k, pair_v in sorted(frequent_pairs_confidence.iteritems(), key=lambda (k, v): (v, k), reverse=True)[:5]:
        print "%s: %s" % (pair_k, pair_v)

# --------------------------------------------
    # Question 2 e

    # we count the number of occurences of the 3-tuples candidates
    candidates = defaultdict(lambda: 0)
    for basket in baskets:
        iter = itertools.product([x for x in frequent_items_dict.keys() if set(basket).issuperset(set([x]))]
                                 , [list(x) for x in frequent_pairs_dict.keys() if set(basket).issuperset(set(x))])
        candidates_in_basket = [sorted([x[0], x[1][0], x[1][1]]) for x in iter]
        candidates_in_basket = set([tuple(sorted(x)) for x in candidates_in_basket if len(set(x)) == 3])
        for it in candidates_in_basket:
            candidates[it] += 1

    # we save the frequent ones
    frequent_3t_dict = dict()
    for t in candidates:
        if candidates[t] >= support:
            frequent_3t_dict[t] = candidates[t]

    # Calculation of confidences
    # (A,B,C): confidence means A | B,C or A knowing B,C
    frequent_3t_confidence = dict()
    for k in frequent_3t_dict.keys():
        frequent_3t_confidence[(k[0], k[1], k[2])] = float(frequent_3t_dict[k]) / frequent_pairs_dict[(k[1], k[2])]
        frequent_3t_confidence[(k[1], k[0], k[2])] = float(frequent_3t_dict[k]) / frequent_pairs_dict[(k[0], k[2])]
        frequent_3t_confidence[(k[2], k[0], k[1])] = float(frequent_3t_dict[k]) / frequent_pairs_dict[(k[0], k[1])]

        frequent_3t_dict = sorted(frequent_3t_dict, key=lambda x:(x[1], x[2]))

    for pair_k, pair_v in sorted(frequent_3t_confidence.iteritems(), key=lambda (k, v): (-v, k[1], k[2]))[:5]:
        print "%s: %s" % (pair_k, pair_v)



