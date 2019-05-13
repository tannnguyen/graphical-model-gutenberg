import numpy as np
import time
import h5py
import argparse
import re
import copy
import os
from nltk import word_tokenize

class LBL:
    def __init__(self, sentences = None, alpha = 0.0001, min_alpha = 0.0001, dim = 100, context = 5, threshold = 3, batches = 1000):
        '''
        vocab, for each word, stores its corresponding namedtuple word
        index2word records the index for each word
        total is the number of words in the training set
        alpha and min_alpha are the upper bound and lower bound for the learning rate
        dim is the dimension for each word embedding
        wordEm is a (vocabulary_size * dim) matrix, each row of which is a word embedding
        context is the size of history window
        words occur less than threshold times will be regarded as rare and will be mapped to a special token '<>'
        <_> is null padding, <s> denotes start of sentence, </s> means the end of sentence
        '''
        self.vocab = {}
        self.index2word = []
        self.frequencies = []
        self.total = -1
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.wordEm = self.contextW = self.biases = None
        self.dim = dim
        self.context = context
        self.threshold = threshold
        self.l_pad = ['<_>'] * (self.context - 1) + ['<s>']
        self.r_pad = ['</s>']
        if sentences is not None:
            self.prepare_vocabulary(sentences)
            self.initialise()
            self.train(sentences, alpha = alpha, min_alpha = min_alpha, batches = batches)

    def initialise(self):
        print('Initialising weights...')
        self.contextW = [(np.random.rand(self.dim, self.dim).astype(np.float32) - 0.5) / self.dim for i in range(self.context) ]
        self.wordEm = (np.random.rand(len(self.vocab), self.dim).astype(np.float32) - 0.5) / self.dim
        self.biases = np.asarray(self.frequencies, np.float32) / np.sum(self.frequencies)
        

    def prepare_vocabulary(self, sentences):
        print('Building vocabulary...')
        total = 0
        vocab = {}
        for sen_no, sentence in enumerate(sentences):
            for w in sentence:
                total += 1
                count = vocab.get(w, 0) + 1
                vocab[w] = count
        self.total = total
        
        self.vocab = {}
        self.index2word = []
        self.frequencies = []
        index = 0
        count_oov = 0
        for w, count in vocab.items():
            if count >= self.threshold:
                self.vocab[w] = index
                self.index2word.append(w)
                self.frequencies.append(count)
                index += 1
            else:
                count_oov += count
        self.vocab['<>'] = index
        index += 1
        self.vocab['<s>'] = index
        index += 1
        self.vocab['</s>'] = index
        self.index2word.extend(['<>', '<s>', '</s>'])
        self.frequencies.extend([count_oov, sen_no, sen_no] )
        print('\nThe size of vocabulary is: {0}, with threshold being {1}\n'.format(len(self.vocab), self.threshold) )


    def train(self, sentences, alpha = 0.001, min_alpha = 0.001, batches = 1000):
        print('Start training...')
        self.alpha = alpha
        self.min_alpha = min_alpha
        count = 0
        start = time.time()
        last_elapsed = 0
        RARE = self.vocab['<>']
        r_hat = np.zeros(self.dim, np.float32)
        delta_c = [np.zeros((self.dim, self.dim), np.float32) for i in range(self.context) ]
        delta_r = np.zeros((len(self.vocab), self.dim), np.float32)
        for sentence in sentences:
            sentence = self.l_pad + sentence + self.r_pad
            for pos in range(self.context, len(sentence) ):
                count += 1
                r_hat.fill(0)
                contextEm = []
                contextW = []
                indices = []
                for i, r in enumerate(sentence[pos - self.context : pos]):
                    if r == '<_>':
                        continue
                    index = self.vocab.get(r, RARE)
                    indices.append(index)
                    ri = self.wordEm[index]
                    ci = self.contextW[i]
                    contextEm.append(ri)
                    contextW.append(ci)
                    r_hat += np.dot(ci, ri)
                energy = np.exp(np.dot(self.wordEm, r_hat) + self.biases)
                probs = energy / np.sum(energy)
                w_index = self.vocab.get(sentence[pos], RARE)
                w = self.wordEm[w_index]

                probs[w_index] -= 1
                temp = np.dot(probs, self.wordEm)
                for i in range(len(contextEm) ):
                    delta_c[self.context - len(contextEm) + i] += np.outer(temp, contextEm[i] )
                VRC = np.zeros(self.dim, np.float32)
                for i in range(len(contextEm) ):
                    VRC += np.dot(contextEm[i], contextW[i].T)
                delta_r += np.outer(probs, VRC)
                for i in range(len(contextEm) ):
                    delta_r[indices[i] ] += np.dot(temp, contextW[i])

                # update after visiting batches sequences
                if count % batches == 0:
                    alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1 - 1.0 * count / self.total)
                    for i in range(self.context):
                        self.contextW[i] -= (delta_c[i] + 1e-5 * self.contextW[i]) * alpha
                    self.wordEm -= (delta_r + 1e-4 * self.wordEm) * alpha
                    for i in range(self.context):
                        delta_c[i].fill(0)
                    delta_r.fill(0)
                elapsed = time.time() - start
                if elapsed - last_elapsed > 1:
                    # print('visited {0} words, with {1:.2f} Ws/s, alpha: {2}.'.format(count, count / elapsed, alpha) )
                    last_elapsed = elapsed

        # add all remaining gradients
        if count % batches != 0:
            alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1 - 1.0 * count / self.total)
            for i in range(self.context):
                self.contextW[i] -= (delta_c[i] + 1e-5 * self.contextW[i]) * alpha
            self.wordEm -= (delta_r + 1e-4 * self.wordEm) * alpha
        print('Training is finished!')


    def perplexity(self, sentences, arpalm=None, weight=None, new_lm=None):
        LOG10TOLOG = np.log(10)
        LOGTOLOG10 = 1. / LOG10TOLOG

        print('Calculating perplexity...')
        RARE = self.vocab['<>']
        # _no_eos means no end of sentence tag </s>
        count_no_eos = count = 0
        logProbs_no_eos = logProbs = 0
        r_hat = np.zeros(self.dim, np.float32)
        for sentence in sentences:
            arpa_sentence = list(reversed(sentence))
            sentence = self.l_pad + sentence + self.r_pad
            for pos in range(self.context, len(sentence)):
                count += 1
                count_no_eos += 1
                r_hat.fill(0)
                for i, r in enumerate(sentence[pos - self.context : pos]):
                    if r == '<_>':
                        continue
                    index = self.vocab.get(r, RARE)
                    ri = self.wordEm[index]
                    ci = self.contextW[i]
                    r_hat += np.dot(ci, ri)
                w_index = self.vocab.get(sentence[pos], RARE)
                energy = np.exp(np.dot(self.wordEm, r_hat) + self.biases)
                res = np.log(energy[w_index] / np.sum(energy) )
                if arpalm and weight:
                    arpa_prob = LOGTOLOG10 * arpalm.prob_list(arpa_sentence)
                    res = res * (1.0 - weight) + arpa_prob * weight
                    if new_lm is not None:
                        new_lm.update_prob_list(res, arpa_sentence)

                logProbs += res
                logProbs_no_eos += res
            logProbs_no_eos -= res
            count_no_eos -=1
            # print results after each sentence
            ppl = np.exp(-logProbs / count)
            # print('count: {0}'.format(count) )
            # print('The perplexity is {0}'.format(ppl) )
        # the following displays the final perplexity
        ppl = np.exp(-logProbs / count)
        ppl_no_eos = np.exp(-logProbs_no_eos / count_no_eos)
        print('The perplexity with eos is {0}'.format(ppl) )
        print('               without eos is {0}'.format(ppl_no_eos) )

def tokenize():	
    # file = open(filename).read()
    # Get current directories
    directory = os.getcwd()

    # Get all the files 
    filepath = os.path.join(directory, 'Gutenberg/txt/')
    files = os.listdir(filepath)
    # Make a dict for authors and titles
    titles = dict()
    text_files = dict()

    # Check every file
    for file in files:
        # Split the author and title
        split = file.split('___')
        try:
            author = split[0]
            title = split[1].split('.')[0]
        except:
            # Not a valid title file
            pass
        if author not in titles:
            titles[author] = []
            text_files[author] = []
            
        text_files[author].append(file)
        titles[author].append(title)

    # Read in all the books from an author
    # Each book is considered a document now. 
    documents = []
    books = text_files['Abraham Lincoln']
    for book in books[2:5]:
        file = os.path.join(filepath, book)
        with open(file) as f:
            data = f.read()
            data = data.replace("[^a-zA-Z#]", "")
            data = data.lower()
            documents.append(data)

    tokenized = []
    # Word tokenization
    for doc in documents:
        tokenized.append(word_tokenize(doc))
    return tokenized

def tokenize_test():
        # file = open(filename).read()
    # Get current directories
    directory = os.getcwd()

    # Get all the files 
    filepath = os.path.join(directory, 'Gutenberg/txt/')
    files = os.listdir(filepath)
    # Make a dict for authors and titles
    titles = dict()
    text_files = dict()

    # Check every file
    for file in files:
        # Split the author and title
        split = file.split('___')
        try:
            author = split[0]
            title = split[1].split('.')[0]
        except:
            # Not a valid title file
            pass
        if author not in titles:
            titles[author] = []
            text_files[author] = []
            
        text_files[author].append(file)
        titles[author].append(title)

    # Read in all the books from an author
    # Each book is considered a document now. 
    books = text_files['Mark Twain']
    book = books[-1]
    file = os.path.join(filepath, book)
    with open(file) as f:
        data = f.read()
        data = data.replace("[^a-zA-Z#]", "")
        data = data.lower()

    tokenized = []
    # Word tokenization
    tokenized.append(word_tokenize(data))
    return tokenized

def train_n_evaluate(filename1, arpa=None, weight=0, new_lm =None):
    sentences = tokenize()
    lm = LBL(sentences)
    lm.perplexity(sentences, arpa, weight, new_lm)
    return lm

def evaluate(filename, lm, arpa=None, weight=0, new_lm=None):
    sentences = tokenize_test()
    lm.perplexity(sentences, arpa, weight, new_lm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument("--train", default=None, help="Train text file")
    parser.add_argument("--save-net", default="lbl.hdf5", dest="save_net")

    # evaluating arguments
    parser.add_argument("--ppl", default=None, 
                        help="Computes PPL of net on text file (if we train, do that after training)")
    parser.add_argument("--net", default=None, 
                        help="Net file to load")
    parser.add_argument("--arpa", metavar="FILE weight", default=None, nargs=2,
                        help="ARPA n-gram model with interpolating, weight as second parameter")
    parser.add_argument('--save-lm', dest='save_lm', metavar="FILE", default=None,
                        help='Saves fixed ARPA language model to file')


    # common
    parser.add_argument("--alg", default="LBL", choices=["LBL", "HLBL", "LBL_MP"],
                        help="Algorithm")

    args = parser.parse_args()

    if args.train:
        print("{0} algorithm training".format(args.alg))
        lm = train_n_evaluate(args.train)

    if args.ppl:
        print("{0} algorithm evaluating".format(args.alg))
        evaluate(args.ppl, lm)