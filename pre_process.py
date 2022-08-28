import os
import json
import nltk
import string
import numpy as np
from tqdm import tqdm


def _get_vocab(all_data):
    """
    Get the vocabulary list.
    """
    vocab_path = 'vocab.json'
    if os.path.exists(vocab_path):
        print('[Pre-Process] Load vocab from vocab.json')
        with open(vocab_path, 'r') as fp:
            vocab = json.load(fp)
            return vocab
    else:
        vocab = {'[pad]': 0, '[unk]': 1}
        max_length = 0
        print('[Pre-Process] Did not find vocab.json')
        print('[Pre-Process] Now generating vocab.json')
        for data in tqdm(all_data):
            text = data['title']
            title_length = 0
            words = nltk.tokenize.word_tokenize(text)
            porter_stemmer = nltk.stem.porter.PorterStemmer()
            for word in words:
                word = word.lower()
                word = porter_stemmer.stem(word)
                word = word.strip(string.digits)
                word = word.strip(string.punctuation)
                if len(word) == 0:
                    continue
                title_length += 1
                if word not in vocab:
                    vocab[word] = len(vocab)
            max_length = max(max_length, title_length)
        # save to json file
        with open(vocab_path, 'w') as fp:
            json.dump(vocab, fp, indent=2)
        print('Max length of title: {}'.format(max_length))
        return vocab


def my_get_vocab(all_data):
    """
    Get the vocabulary list.
    """
    vocab_path = 'vocab.json'
    if os.path.exists(vocab_path):
        print('[Pre-Process] Load vocab from vocab.json')
        with open(vocab_path, 'r') as fp:
            vocab = json.load(fp)
            return vocab
    else:
        vocab_count = {}
        max_length = 0
        print('[Pre-Process] Did not find vocab.json')
        print('[Pre-Process] Now generating vocab.json')
        for data in tqdm(all_data):
            text = data['title']
            stopwords = set(nltk.corpus.stopwords.words('english'))
            title_length = 0
            words = nltk.tokenize.word_tokenize(text)
            porter_stemmer = nltk.stem.porter.PorterStemmer()
            for word in words:
                word = word.lower()
                word = porter_stemmer.stem(word)
                word = word.strip(string.digits)
                word = word.strip(string.punctuation)
                if len(word) == 0 or word in stopwords:
                    continue
                title_length += 1
                if word not in vocab_count:
                    vocab_count[word] = 1
                else:
                    vocab_count[word] += 1
            max_length = max(max_length, title_length)
        vocab_count = [(word, vocab_count[word]) for word in vocab_count]
        vocab_count.sort(key=lambda entry: entry[1], reverse=True)
        max_word_num = 1000
        vocab = {'[pad]': 0, '[unk]': 1}
        for i in range(max_word_num):
            vocab[vocab_count[i][0]] = i + 2
        # save to json file
        with open(vocab_path, 'w') as fp:
            json.dump(vocab, fp, indent=2)
        print('Max length of title: {}'.format(max_length))
        return vocab


def _wrap_dataset(all_data):
    """
    Generate numpy array format dataset.
    """
    x, y = [], []
    cate_map = {
        'b': 0,
        't': 1,
        'e': 2,
        'm': 3,
    }
    for data in tqdm(all_data):
        y.append(cate_map[data['category']])
        x.append(data['rep'])
    x = np.array(x)
    y = np.array(y)
    return (x, y)


def to_bag_of_words(all_data):
    """
    "Bag Of Words" method.
    """
    bow_path_x = 'bow_x.npy'
    bow_path_y = 'bow_y.npy'
    if os.path.exists(bow_path_x) and os.path.exists(bow_path_y):
        print('[Pre-Process] Load BOW from local files')
        x = np.load(bow_path_x)
        y = np.load(bow_path_y)
        return (x, y)
    
    vocab = my_get_vocab(all_data)
    vocab_size = len(vocab)
    print('Generating BOW representation...')
    for data in tqdm(all_data):
        # to lowercase
        text = data['title']
        data['rep'] = [0 for _ in range(vocab_size)]
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words = nltk.word_tokenize(text)
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        for word in words:
            word = word.lower()
            word = porter_stemmer.stem(word)
            word = word.strip(string.digits)
            word = word.strip(string.punctuation)
            if len(word) == 0 or word in stopwords or word not in vocab:
                continue
            idx = vocab[word]
            data['rep'][idx] += 1
    
    x, y = _wrap_dataset(all_data)
    print('[Pre-Process] Save BOW to local files')
    np.save(bow_path_x, x)
    np.save(bow_path_y, y)
    return (x, y)


def to_id_list(all_data):
    """
    "ID List" method.
    """
    MAX_LENGTH = 21
    idlist_path_x = 'idlist_x.npy'
    idlist_path_y = 'idlist_y.npy'
    if os.path.exists(idlist_path_x) and os.path.exists(idlist_path_y):
        print('[Pre-Process] Load id-list from local files')
        x = np.load(idlist_path_x)
        y = np.load(idlist_path_y)
        return (x, y)

    vocab = _get_vocab(all_data)
    print('Generating id-list representation...')
    for data in tqdm(all_data):
        # to lowercase
        text = data['title']
        data['rep'] = []
        words = nltk.word_tokenize(text)
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        for word in words:
            word = word.lower()
            word = porter_stemmer.stem(word)
            word = word.strip(string.digits)
            word = word.strip(string.punctuation)
            if len(word) == 0:
                continue
            data['rep'].append(vocab[word])
        # padding
        data['rep'] += [0 for _ in range(len(data['rep']), MAX_LENGTH)]
    
    x, y = _wrap_dataset(all_data)
    print('[Pre-Process] Save id-list to local files')
    np.save(idlist_path_x, x)
    np.save(idlist_path_y, y)
    return (x, y)


def to_word_vector(all_data, glove_dir='./glove', cache=True):
    """
    "word2vec" method.
    """
    MAX_LENGTH = 21
    WORDVEC_DIM = 300
    wordvec_path_x = 'wordvec_x.npy'
    wordvec_path_y = 'wordvec_y.npy'
    if cache and os.path.exists(wordvec_path_x) and os.path.exists(wordvec_path_y):
        print('[Pre-Process] Load word vectors from local files')
        x = np.load(wordvec_path_x)
        y = np.load(wordvec_path_y)
        return (x, y)

    word2id_file = os.path.join(glove_dir, 'word2id.txt')
    with open(word2id_file, 'r', encoding='utf-8') as fp:
        word_list = eval(fp.read())
    word2id = {}
    for i, word in enumerate(word_list):
        word2id[word] = i

    word2vec_file = os.path.join(glove_dir, 'word2vec.npy')
    word2vec = np.load(word2vec_file)

    print('Generating word vector representation...')
    for data in tqdm(all_data):
        text = data['title']
        data['rep'] = np.zeros((MAX_LENGTH, WORDVEC_DIM))
        words = nltk.word_tokenize(text)
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        count = 0
        for word in words:
            word = word.lower()
            word = porter_stemmer.stem(word)
            word = word.strip(string.digits)
            word = word.strip(string.punctuation)
            if len(word) == 0:
                continue
            if word not in word2id:
                # ignore the unk
                continue
            word_id = word2id[word]
            data['rep'][count] = word2vec[word_id]
            count += 1
        data['rep'] = np.sum(data['rep'], axis=0)

    x, y = _wrap_dataset(all_data)
    print('[Pre-Process] Save word vectors to local files')
    np.save(wordvec_path_x, x)
    np.save(wordvec_path_y, y)
    return (x, y)


def to_tokenized(all_data):
    """
    Use 'tokenizer' from transformers
    """
    pass


def split_dataset(dataset, test_ratio=0.1, val_ratio=0.05):
    """
    Split originial dataet into training set, validation set and test set.
    """
    x, y = dataset
    assert len(x) == len(y)
    size = len(x)
    test_size = int(size * test_ratio)
    val_size = int(size * val_ratio)
    train_size = size - test_size - val_size

    train_set = (x[:train_size], y[:train_size])
    val_end = train_size + val_size
    val_set = (x[train_size: val_end], y[train_size: val_end])
    test_set = (x[val_end:], y[val_end:])

    return (train_set, val_set, test_set)


if __name__ == '__main__':
    from load_data import load_data
    all_data = load_data()
    dataset = split_dataset(to_bag_of_words(all_data[:1000]))
    dataset = split_dataset(to_id_list(all_data[:1000]))
    train_set, val_set, test_set = dataset
