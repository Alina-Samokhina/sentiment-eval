import numpy as np
import pandas as pd
import re

import pymorphy2
from nltk import word_tokenize
import xml.etree.ElementTree as et 

def read_restaurant_dataset(file):
    '''Reads restaurant reviews xml from SentiRuEval-2015
    params:
    :file: path to xml with data
    returns intermediate dataset as pd.DataFrame
    '''
    categories = ['Food', 'Interior', 'Price', 'Whole', 'Service']    
    dataset = pd.DataFrame(columns = ['review_id', 
                                          'obj', 'user', 'date', 'useful', 
                                          'food_score', 'interior_score', 'service_score',
                                      'text', 
                                      'Food', 'Interior', 'Price', 'Whole', 'Service'])
    xtree = et.parse(file)
    xroot = xtree.getroot()
    for node in xroot:
        row = {}
        row['review_id'] = node.attrib['id']
        row['obj'] = (node.find('meta').find('object').text)
        row['user'] = (node.find('meta').find('user').text)
        row['date'] = (node.find('meta').find('date').text)
        row['useful'] = (node.find('meta').find('useful').text)
        row['text'] = (node.find('text').text)
        for score in ['food', 'interior', 'service']:
            row[score+'_score'] = (node.find('scores').find(score).text)
        cats = node.find('categories').findall('category')
        sentiments = {}
        for i in range(len(cats)):
            sentiments[categories[i]] = cats[i].get('sentiment')
        row.update(sentiments)
        dataset = dataset.append(row, ignore_index = True)
    return dataset

def read_car_dataset(file):
    '''Reads car reviews xml from SentiRuEval-2015
    params:
    :file: path to xml with data
    returns intermediate dataset as pd.DataFrame
    '''
    categories = ['Comfort', 'Appearance', 'Reliability', 'Safety', 'Driveability', 'Whole', 'Costs']
    dataset = pd.DataFrame(columns = ['review_id', 'obj', 'text', 'Comfort', 'Appearance', 'Reliability', 'Safety', 'Driveability', 'Whole', 'Costs'])
    xtree = et.parse(file)
    xroot = xtree.getroot()
    for node in xroot:
        row = {}
        row['review_id'] = node.attrib['id']
        row['obj'] = (node.find('meta').find('object').text)
        row['text'] = (node.find('text').text)
        cats = node.find('categories').findall('category')
        sentiments = {}
        for i in range(len(cats)):
            sentiments[categories[i]] = cats[i].get('sentiment')
        row.update(sentiments)
        dataset = dataset.append(row, ignore_index = True)
    return dataset

def read_aspects(file):
    '''Extracts apects rom xml of SentiRuEval-2015'''
    dataset = pd.DataFrame(columns = ['asp_id', 'review_id', 
                                      'from', 'mark', 'sentiment', 'term', 'to', 'type', 'category'])
    xtree = et.parse(file)
    xroot = xtree.getroot()
    asp_id = 0
    for node in xroot:
        asps = node.find('aspects').findall('aspect')
        properties = ['from', 'mark', 'sentiment', 'term', 'to', 'type', 'category']
        for asp in asps:
            row = {}
            row['review_id'] = node.attrib['id']
            row['asp_id'] = asp_id
            aspect = {}
            for prop in properties:
                aspect[prop] = asp.get(prop)
            asp_id += 1
            row.update(aspect)
            dataset = dataset.append(row, ignore_index = True)
    return dataset

def parse_dataset(xml_file, ds_type='rest'):
    '''Parse xml file about restaurant or cars from SentiRuEval2015
    into pd.DataFrame for most of the tasks
    params:
    :xml_file: path to xml
    :ds_type: type of reviews in xml, 'car' or 'rest'. default - restaurants 
    Returns:
        dataframe
    '''
    if ds_type == 'rest':
        texts_df = read_restaurant_dataset(xml_file)
    elif ds_type == 'car':
        texts_df == read_car_dataset(xml_file)
    
    aspects_df = read_aspects(xml_file)
    
    texts_df = texts_df[['review_id','text']]
    aspects_df = aspects_df[['review_id', 'term', 'sentiment', 'category', 'from', 'to']][aspects_df['type']=='explicit'][aspects_df['mark']=='Rel']    
    dataset = pd.DataFrame(columns = ['review', 'text', 'aspects', 'sentiments', 'categories'])
    for review in texts_df.iterrows():
        review_id = review[1]['review_id']
        text = review[1]['text']
        aspects = [asp for asp in aspects_df['term'][aspects_df['review_id']==review_id]]
        sentiments = [sent for sent in aspects_df['sentiment'][aspects_df['review_id']==review_id]]
        categories = [cat for cat in aspects_df['category'][aspects_df['review_id']==review_id]]
        ends = [st for st in aspects_df['to'][aspects_df['review_id']==review_id]]
        starts = [st for st in aspects_df['from'][aspects_df['review_id']==review_id]]
        assert(len(aspects)==len(sentiments)==len(categories))
        row = {
            'review': review_id,
            'text': text,
            'aspects': aspects,
            'sentiments': sentiments,
            'categories': categories,
            'starts': starts,
            'ends': ends
        }
        dataset = dataset.append(row, ignore_index = True)
    return dataset

def tokenize_text(text):
    text = re.sub(r"[\"\'\\/,....;:@#?!&$-]+", ' ', text)  
    text = re.sub(r"\s+", ' ', text)
    text =  word_tokenize(text)
    return text


def get_word_poses(tok_text, text):
    '''Turns word numbers in their positions in chars
    params:
        :tok_text: tokenized text
        :text: raw text
    returns:
        arrays of positions in text
    '''
    last_index = 0
    words_pos = []
    for word in tok_text:
        idxs = text.find(word, last_index)
        idxe = idxs+len(word)
        words_pos.append(idxs)
        last_index = idxs+1
    return np.array(words_pos)

def get_mask(row):
    '''Gets 0, 1, 2 for non-aspect/aspect/part-of-aspect word in text
    params:
        :row: row of a dataframe
    returns:
        np.array of aspect "class"
    '''
    text = row['text']
    aspects = row['aspects']
    pos_from = row['starts']
    pos_to = row['ends']
    

    tokenized_text = (tokenize_text(text))
    mask = np.zeros(len(tokenized_text))
    
    
    words_pos = get_word_poses(tokenized_text, text)
    for asp, st, end in zip(aspects, pos_from, pos_to):
        indices = np.where((words_pos>=int(st)) & (words_pos<=int(end)))[0]
        if len(indices) == 1:
            mask[indices] = 1
        else:
            for i, ind in enumerate(indices):
                mask[ind] = min(i, 2)
    return mask

def get_dataset(xml_file, ds_type='rest'):
    '''Gets dataset for using in different tasks
    params:
        :xml_file: path to xml
        :ds_type: type of reviews in xml, 'car' or 'rest'. default - restaurants 
    Returns:
        dataframe
    '''
    ds = parse_dataset(xml_file, ds_type)
    masks = []
    for row in ds.iterrows():
        mask = get_mask(row[1])
        masks.append(mask)
    ds['mask_asp'] = masks
    ds['text'] = ds['text'].apply(tokenize_text)
    morph = pymorphy2.MorphAnalyzer()
    ds['text'] = [[morph.parse(w)[0].normal_form for w in ds.text[k]] for k in range(len(ds.text))]
    return ds[['text', 'aspects', 'categories', 'sentiments', 'mask_asp']]

def get_ds_vocab(ds):
    words = sorted(set(np.hstack(np.array(ds['text'].values))))
    return words

def get_sentences(ds):
    return ds.text.values

def text2ind(ds, vocab):
    '''
    Turns text fiels of dataset into indexes corresponding to vocabulary
    params:
        :ds: dataset
        :vocab: vocabulary to use
    returns:
        altered dataset
    '''
    dataset = ds.copy()
    dataset['text'] = [[vocab[word] if (word in vocab.keys()) else vocab['<UNK>'] for word in dataset['text'][k]] for k in range(len(dataset.text))]
    return dataset

