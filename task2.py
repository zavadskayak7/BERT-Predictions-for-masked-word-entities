### IMPORTS

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lama.modules.bert_connector import *
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics
import io
import jsonlines

### FUNCTIONS

def write_test_prediction(file_name, map_idx_to_id, label):
    ''' function to save test predictions into jsonl file
    '''
    fp = io.BytesIO()
    with jsonlines.open(file_name, mode='w') as writer:
        for key, val in map_idx_to_id.items():
            l = label[key]
            if l:
                pred = {'id':val,'label':'SUPPORTS'}
            else:
                pred = {'id':val,'label':'REFUTES'}
            writer.write(pred)
    writer.close()
    fp.close()

def data_label_split(file, model, train_set=True):
    #common_vocab = load_vocab("/content/bert/cased_L-12_H-768_A-12/vocab.txt")
    data, label = [], []
    idx = 0
    map_idx_to_id = {}
    with jsonlines.open(file) as reader:
        for obj in reader:
            
            # mask entity
            text = [obj['claim'][:obj['entity']['start_character']] + '[MASK]' + obj['claim'][obj['entity']['end_character']:]]
            # get contextual embedding of masked line
            contextual_embeddings_mask, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings([text])
            # get index of the token [MASK]
            mask_idx = tokenized_text_list[0].index('[MASK]')
            # transform to array emb of masked
            emb_of_masked = np.asarray(contextual_embeddings_mask[11][0][mask_idx])
            # get contextual embedding of original line
            contextual_embeddings_mask_orig, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings([[obj['claim']]])
            # transform to array emb of original entity
            emb_of_entity = np.asarray(contextual_embeddings_mask_orig[11][0][mask_idx])
            # concatenate embeddings
            data.append(np.concatenate((emb_of_masked, emb_of_entity), axis=0))

            if train_set:
                if obj['label'] == 'SUPPORTS':
                    label.append(1)
                else:
                    label.append(0)
            else:
                map_idx_to_id[idx] = obj['id']
                
            idx += 1
    return np.asarray(data), np.asarray(label), map_idx_to_id

### MAIN

class BertArgs:
  def __init__(self):
    self.bert_model_dir = "/content/drive/My Drive/DMT_HW3/bert/cased_L-12_H-768_A-12"
    self.bert_model_name = "bert-base-cased"
    self.bert_vocab_name = "vocab.txt"

args = BertArgs()
model = Bert(args)

# split train and dev data
train_file = '/content/drive/My Drive/DMT_HW3/train_fever.json'
dev_file = '/content/drive/My Drive/DMT_HW3/dev_fever.json'
data_train, label_train, map_idx_to_id_train = data_label_split(train_file, model)
data_dev, label_dev, map_idx_to_id_dev = data_label_split(dev_file, model)

# train model
clf = LinearDiscriminantAnalysis(solver='svd')
# fit model
clf.fit(data_train, label_train)
# get acc
LDA_acc = clf.score(data_dev, label_dev) # result: 0.6763485477178424
print(LDA_acc)

# get predictions and write to a file
test_file = '/content/drive/My Drive/DMT_HW3/singletoken_test_fever_homework_NLP.jsonl'
data_test, no_label_test, map_idx_to_id = data_label_split(test_file, model, train_set=False)
label_test = clf.predict(data_test)
write_test_prediction('final_test_pred.jsonl',map_idx_to_id,label_test)
