import jsonlines
import io
import pickle
from flair.models import SequenceTagger
from flair.data import Sentence

###
### FUNCTIONS ###
###
def Jsonl(data, file):
    ''' function to save jsonl file after preprocessing
    '''
  f = io.BytesIO()
  with jsonlines.open(file, mode='w') as writer:
    writer.write_all(data)
  writer.close()
  f.close()


def fever_parser(file_name,bert_vocab_list):
    ''' Parsing of train and dev datasets. leaves only sentences with 'SUPPORTS' and 'REFUTES' labels.
    checks only single entities and no spaces, no minus signs
    '''
    train_test_fever = []
    tagger = SequenceTagger.load('ner')
    labels = ['SUPPORTS','REFUTES']

    with jsonlines.open(file_name) as file:
        for line in file:
            if line['label'] in labels:
                sentence = Sentence(line['claim'])
                # predict NER tags
                tagger.predict(sentence)
                ner_predict = sentence.to_dict(tag_type='ner')

                single_token_in_bert = [] # to save a list of the appropriate tokens, should be only one, if more-reject
                # check each entity
                for entity in ner_predict['entities']:
                    text = entity['text']
                    if (' ' in text) or ('-' in text):
                        continue
                    else:
                        if text not in bert_vocab_list:
                            continue
                        else:
                            single_token_in_bert.append(text)
                # continue only if single entity
                if (len(single_token_in_bert) > 1) or (len(single_token_in_bert) == 0):
                    continue
                else:
                    # attach info if the used entity
                    for en in ner_predict['entities']:
                        if en['text'] == single_token_in_bert[0]:
                            datapoint = {'id':line['id'], 'label':line['label'],
                                         'claim':line['claim'],
                                         'entity':{'mention':single_token_in_bert[0],
                                                   'start_character':en['start_pos'],
                                                   'end_character':en['end_pos']}}
                            train_test_fever.append(datapoint)
                            #print(datapoint)
    return train_test_fever

###
### MAIN ###
###

main_path = '/content/drive/My Drive/DMT_HW3'
bert_vocab = '/content/drive/My Drive/DMT_HW3/cased_L-24_H-1024_A-16/vocab.txt'

### READ AND SAVE BERT VOCAB ###
# create bert vocab list
bert_vocab_list = []
with open(bert_vocab) as f:
    for w in f:
        bert_vocab_list.append(w[:-1])
# save bert vocab list
with open(main_path+'/bert_vocab_list'+'.pickle', 'wb') as f:
            pickle.dump(bert_vocab_list, f)
# to read pickle with open(main_path+'/bert_vocab_list'+'.pickle', 'rb') as f: bert_vocab_list = pickle.load(f)



### READ AND PARSE DEV SET ###
dev_set_name = '/content/drive/My Drive/DMT_HW3/paper_dev.jsonl'
# save parsed dev dataset
dev_fever = fever_parser(dev_set_name,bert_vocab_list)
with open(main_path+'/dev_fever'+'.pickle', 'wb') as f:
            pickle.dump(dev_fever, f)
Jsonl(dev_fever,main_path+'/dev_fever'+'.json')


### READ AND PARSE TRAIN SET ###
file_name = '/content/drive/My Drive/DMT_HW3/train.jsonl'
dev_set_name = '/content/drive/My Drive/DMT_HW3/paper_dev.jsonl'
# save parsed fever dataset
train_fever = fever_parser(file_name,bert_vocab_list)
with open(main_path+'/train_fever'+'.pickle', 'wb') as f:
            pickle.dump(train_fever, f)
Jsonl(train_fever,main_path+'/train_fever'+'.json')

