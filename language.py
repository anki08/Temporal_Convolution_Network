import string

import torch

import utils
from models import LanguageModel, load_model


def log_likelihood(model: LanguageModel, some_text: str):
    one_hot = utils.one_hot(some_text)
    output = model.predict_all(some_text)
    output = output[:, :output.shape[1] - 1]
    idxs = torch.nonzero(one_hot == 1)
    out = output[idxs[:, 0], idxs[:, 1]]
    ll = out.sum()
    return ll


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100,
                average_log_likelihood: bool = False):
    substring = ""
    beams_sum = list()
    beams_avg = list()
    array1 = list()
    vocab = string.ascii_lowercase + ' .'
    char_set = list(vocab)
    if (beam_size > 28):
        output1 = model.predict_all(substring)
        for i in range(len(output1)):
            char = char_set[i]
            array1.append((char, output1[i]))
        beams_sum = array1
        beams_avg = array1
    loop_size = len(array1)

    array_period_avg = list()
    array_period_sum = list()
    for length in range(max_length):
        array_llsum = list()
        array_llavg = list()
        for k in range(loop_size):
            if (average_log_likelihood == True):
                if (beams_avg[k][0][-1] == '.'):
                    array_period_avg.append((beams_avg[k][0], beams_avg[k][1]))
                    continue
                output = model.predict_all(beams_avg[k][0][-1])
            elif (average_log_likelihood == False):
                if (beams_sum[k][0][-1] == '.'):
                    array_period_sum.append((beams_sum[k][0], beams_sum[k][1]))
                    continue
                output = model.predict_all(beams_sum[k][0][-1])

            char_prob = output[:, -1]
            for i in range(len(char_prob)):
                char = char_set[i]
                if (average_log_likelihood == False):
                    substring = beams_sum[k][0] + char
                    llsum = log_likelihood(model, substring)
                    array_llsum.append((substring, llsum))
                    if (char == '.'):
                        array_period_sum.append((substring, llsum))

                elif (average_log_likelihood == True):
                    substring = beams_avg[k][0] + char
                    llavg = log_likelihood(model, substring) / len(beams_avg[k][0])
                    array_llavg.append((substring, llavg))
                    if (char == '.'):
                        array_period_avg.append((substring, llavg))

        if (average_log_likelihood == False):
            seen = set()
            beams_sum = array_llsum
            beams_sum.extend(array_period_sum)
            beams_sum = [(a, b) for a, b in beams_sum if not (a in seen or seen.add(a))]
            beams_sum = sort_tuple(beams_sum)
            beams_sum = beams_sum[:beam_size]
            loop_size = len(beams_sum)

        elif (average_log_likelihood == True):
            seen = set()
            beams_avg = array_llavg
            beams_avg.extend(array_period_avg)
            beams_avg = [(a, b) for a, b in beams_avg if not (a in seen or seen.add(a))]
            beams_avg = sort_tuple(beams_avg)
            beams_avg = beams_avg[:beam_size]
            loop_size = len(beams_avg)

    if (average_log_likelihood == True):
        seen = set()
        beams_avg.extend(array_period_avg)
        beams_avg = [(a, b) for a, b in beams_avg if not (a in seen or seen.add(a))]
        beams_avg = sort_tuple(beams_avg)
        substring = [i[0] for i in beams_avg]

    elif (average_log_likelihood == False):
        seen = set()
        beams_sum.extend(array_period_sum)
        beams_sum = [(a, b) for a, b in beams_sum if not (a in seen or seen.add(a))]
        beams_sum = sort_tuple(beams_sum)
        substring = [i[0] for i in beams_sum]

    return substring[:n_results]


def sort_tuple(tup):
    return (sorted(tup, key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = load_model()

    for s in beam_search(lm, 100):
        print(s)
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s)
