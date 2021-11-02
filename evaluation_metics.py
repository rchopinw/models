import numpy as np
from collections import Counter
from itertools import chain, product
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer


# calculating WER (word error rate)
def wer(reference, hypothesis):
    reference = reference.lower().split()
    hypothesis = hypothesis.lower().split()
    rl, hl = len(reference), len(hypothesis)
    dp = np.zeros(
        shape=(rl + 1, hl + 1)
    )
    for i in range(rl + 1):
        dp[i, 0] = i
    for j in range(hl + 1):
        dp[0, j] = j
    for i in range(1, rl + 1):
        for j in range(1, hl + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)
    return dp[-1][-1] / len(reference)


# calculating BLEU (Bilingual Evaluation Understudy)
class BLEU:
    def __init__(self, n, weights=None):
        if weights is None:
            self.weights = np.array([1/n for _ in range(n)]).reshape((n, 1))
        else:
            self.weights = np.array(weights).reshape((n, 1))
        self.n = n

    def __n_gram_generator(self, s, n):
        s = s.lower().split()
        word_list = [' '.join(s[i - n : i]) for i in range(len(s) + 1) if i >= n]
        return word_list

    def __nist_length_penalty(self, ref_len, hyp_len):
        ratio = hyp_len / ref_len
        if 0 < ratio < 1:
            ratio_x, score_x = 1.5, 0.5
            beta = np.log(score_x) / np.log(ratio_x) ** 2
            return np.exp(beta * np.log(ratio) ** 2)
        else:  # ratio <= 0 or ratio >= 1
            return max(min(ratio, 1.0), 0.0)

    def cal_bleu_score(self, reference, hypothesis):
        r_l, h_l = len(reference.split()), len(hypothesis.split())

        b = 1 if h_l > r_l else np.exp(1 - h_l / r_l)
        clipped_precision_scores = np.zeros(
            shape=(1, self.n)
        )

        for i in range(1, 1 + self.n):
            r_ngram = Counter(
                self.__n_gram_generator(
                    reference, i
                )
            )
            h_ngram = Counter(
                self.__n_gram_generator(
                    hypothesis, i
                )
            )

            c = sum(h_ngram.values())

            for gram in h_ngram:
                if gram in r_ngram:
                    if h_ngram[gram] > r_ngram[gram]:
                        h_ngram[gram] = r_ngram[gram]
                else:
                    h_ngram[gram] = 0

            clipped_precision_scores[0, i-1] = sum(h_ngram.values()) / c

        return b * np.exp(
            np.log(clipped_precision_scores) @ self.weights
        )

    def cal_nist_score(self, reference, hypothesis):
        ngram_freq = Counter()
        total_ref_words = 0
        for ref in reference:
            for w in ref:
                for i in range(1, self.n + 1):
                    ngram_freq.update(
                        self.__n_gram_generator(w, i)
                    )
                total_ref_words += len(w)
        information_weights = {}
        for gram in ngram_freq:
            gram = gram[::-1]
            if gram and gram in ngram_freq:
                numerator = ngram_freq[gram]
            else:
                numerator = total_ref_words
            information_weights[gram] = np.log(numerator / ngram_freq[gram], 2)
        nist_precision_numerator_gram = Counter()
        nist_precision_denominator_gram = Counter()
        l_ref, l_sys = 0, 0
        for i in range(1, self.n + 1):
            for r, h in zip(reference, hypothesis):
                h_l = len(h)
                nist_score_per_ref = []
                for ref in r:
                    r_l = len(ref)
                    hyp_ngrams = (
                        Counter(
                            self.__n_gram_generator(
                                h, i
                            ) if len(h) >= 1
                            else Counter()
                        )
                    )
                    ref_ngrams = (
                        Counter(
                            self.__n_gram_generator(
                                ref, i
                            ) if len(r) >= i
                            else Counter()
                        )
                    )
                    ngram_overlaps = hyp_ngrams & ref_ngrams
                    numerator = sum(
                        information_weights[gram] * count
                        for gram, count in ngram_overlaps.items()
                    )
                    denominator = sum(hyp_ngrams.values())
                    precision = 0 if denominator == 0 else numerator / denominator
                    nist_score_per_ref.append(
                        (precision, numerator, denominator, r_l)
                    )
                precision, numerator, denominator, r_l = max(nist_score_per_ref)
                nist_precision_numerator_gram[i] += numerator
                nist_precision_denominator_gram[i] += denominator
                l_ref += r_l
                l_sys += h_l
        nist_precision = 0
        for i in nist_precision_numerator_gram:
            precision = (
                nist_precision_numerator_gram[i]
                / nist_precision_denominator_gram[i]
            )
            nist_precision += precision
        return nist_precision * self.__nist_length_penalty(l_ref, l_sys)

# Meteor Score
class MeteorScore:
    def __init__(self, stem=PorterStemmer, preprocess=str.lower):
        self.stemmer = stem
        self.preprocess = preprocess

    def __generate_enumerates(self, hypothesis, reference):
        reference = [(i, x)
                     for i, x in enumerate(
                self.preprocess(
                    reference
                ).split()
            )]
        hypothesis = [(i, x)
                      for i, x in enumerate(
                self.preprocess(
                    hypothesis
                ).split()
            )]
        return reference, hypothesis

    def __match_enumerates(self, hypothesis, reference):
        word_match = []
        for hid, h in hypothesis[::-1]:
            for rid, r in reference[::-1]:
                if h == r:
                    word_match.append(
                        (hid, rid)
                    )
                    hypothesis.pop(hid)
                    reference.pop(rid)
                    break
        return word_match, hypothesis, reference

    def __match_stem_enumerates(self, hypothesis, reference):
        stem_hypo = [
            (hid, self.stemmer(h)) for hid, h in hypothesis
        ]
        stem_ref = [
            (rid, self.stemmer(r)) for rid, r in reference
        ]
        return self.__match_enumerates(
            stem_hypo, stem_ref
        )

    def __match_wordsyn_enumerates(self, hypothesis, reference, wordnet=wordnet):
        word_match = []
        for hid, h in hypothesis[::-1]:
            hypo_syns = set(
                chain.from_iterable(
                    (
                        lemma.name()
                        for lemma in synset.lemmas()
                        if lemma.name().find('_') < 0
                    )
                    for synset in wordnet.synsets(h)
                )
            ).union({h})
            for rid, r in reference[::-1]:
                if r in hypo_syns:
                    word_match.append(
                        (hid, rid)
                    )
                    hypothesis.pop(hid)
                    reference.pop(rid)
                    break
        return word_match, hypothesis, reference

    def __align_words_enumerates(self, hypothesis, reference):
        exact_matches, hyp, ref = self.__match_enumerates(
            hypothesis, reference
        )
        stem_matches, hyp, ref = self.__match_stem_enumerates(
            hypothesis, reference
        )
        wns_matches, hyp, ref = self.__match_wordsyn_enumerates(
            hypothesis, reference
        )
        return (
            sorted(
                exact_matches + stem_matches + wns_matches, key=lambda x: x[0]
            ),
            hyp,
            ref,
        )

    def __count_chunks(self, matches):
        i = 0
        chunks = 1
        while i < len(matches) - 1:
            if (matches[i + 1][0] == matches[i][0] + 1) and (
                    matches[i + 1][1] == matches[i][1] + 1
            ):
                i += 1
                continue
            i += 1
            chunks += 1
        return chunks

    def exact_match(self, hypotheses, reference):
        return self.__match_enumerates(
            *self.__generate_enumerates(
                hypotheses, reference
            )
        )

    def stem_match(self, hypothesis, reference):
        return self.__match_stem_enumerates(
            *self.__generate_enumerates(
                hypothesis, reference
            )
        )

    def wordnetsyn_match(self, hypothesis, reference):
        return self.__match_wordsyn_enumerates(
            *self.__generate_enumerates(
                hypothesis, reference
            )
        )

    def single_meteor_score(self, reference, hypothesis, alpha, beta, gamma):
        hyp, ref = self.__generate_enumerates(
            hypothesis, reference
        )
        hyp_len, ref_len = len(hyp), len(ref)
        matches, _, _ = self.__align_words_enumerates(
            hyp, ref
        )
        matches_count = len(matches)
        try:
            precision = float(matches_count) / hyp_len
            recall = float(matches_count) / ref_len
            fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
            chunk_count = float(self.__count_chunks(matches))
            frag_frac = chunk_count / matches_count
        except ZeroDivisionError:
            return 0.0
        return (1 - gamma * frag_frac**beta) * fmean


# TER Score


# GTM Score