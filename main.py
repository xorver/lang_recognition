#!/usr/bin/env python

import os
import numpy
from numpy import numarray


def extract_all_chars(filenames):
    char_set = set()
    for filename in filenames:
        with open(filename) as file:
            for word in file.read().split():
                for char in word.lower():
                    char_set.add(char)
    return char_set


def create_base(char_set, n):
    if n == 1:
        return map(lambda char: (char,), list(char_set))
    new_base = []
    for ngram in create_base(char_set, n-1):
        for char in list(char_set):
            new_base.append(ngram + (char,))
    return new_base


def create_vector_for_files(ngram_base_index, n, filenames):
    vector = [0] * len(ngram_base_index)
    for filename in filenames:
        with open(filename) as file:
            update_vector(ngram_base_index, n, file.read(), vector)
    return vector


def update_vector(ngram_base_index, n, text, vector):
    for string in text.lower().split():
        chars = list(string)
        for i in range(len(chars) - n + 1):
            ngram = tuple(chars[i: i+n])
            try:
                base_index = ngram_base_index[ngram]
                vector[base_index] += 1
            except Exception as e:
                pass
    return vector


def quadratic_euclidean_distance(ngram1, ngram2):
    vec1 = numpy.array(list(ngram1))
    vec2 = numpy.array(list(ngram2))
    vec1 = vec1 / float(numpy.linalg.norm(vec1))
    vec2 = vec2 / float(numpy.linalg.norm(vec2))
    return sum([(vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))])

def cosinus_distance(ngram1, ngram2):
    vec1 = numpy.array(list(ngram1))
    vec2 = numpy.array(list(ngram2))
    return 1 - (sum([(vec1[i] * vec2[i]) for i in range(len(vec1))])) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))


# const
n = 4
ignored_chars = {' ', '$', '(', ',', '.', ':', ';', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\', '`', '\'',
                 '+', '-', '*', '/', '<', '>', '^', '%', '=', '?', '!', '[', ']', '{', '}', '_', '\n', '"', '&', '~'}

# prepare all chars and languages from samples
chars = set()
languages = []
lang_sample_files = []
for dirname, dirnames, filenames in os.walk('samples'):
    languages = languages + dirnames
    if filenames:
        lang_sample_files.append([os.path.join(dirname, filename) for filename in filenames])
        chars = chars | extract_all_chars(lang_sample_files[-1])
chars = chars - ignored_chars

# prepare base
base = create_base(chars, n)
ngram_base_index = {}
for i in range(len(base)):
    ngram_base_index[base[i]] = i
print("Base vector prepared")

# prepare language vectors in given base
lang_vectors = map(lambda filenames: create_vector_for_files(ngram_base_index, n, filenames), lang_sample_files)
print("Language vectors prepared")

# test files
for dirname, dirnames, filenames in os.walk('input'):
    for filename in filenames:
        print(filename + ":")
        in_vector = create_vector_for_files(ngram_base_index, n, [os.path.join(dirname, filename)])
        for i in range(len(languages)):
            print('   ' + languages[i] + ': ' + str(quadratic_euclidean_distance(in_vector, lang_vectors[i]))) + "| " + \
            str(cosinus_distance(in_vector, lang_vectors[i]))