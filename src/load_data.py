import os
from tqdm import tqdm


def load_data(dir_path='../NewsAggregatorDataset'):
    all_data = []
    # Only use newsCorpora.csv for now
    # FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
    filepath = os.path.join(dir_path, 'newsCorpora.csv')
    print('[Load Data] Load data from newsCorpora.csv')
    with open(filepath, 'r') as fp:
        for line in tqdm(fp.readlines()):
            line = line.split('\t')
            title = line[1]
            url = line[2]
            category = line[4]
            all_data.append({
                'title': title,
                'url': url,
                'category': category,
            })
    return all_data


if __name__ == '__main__':
    all_data = load_data()
    print('{} samples in total.'.format(len(all_data)))
