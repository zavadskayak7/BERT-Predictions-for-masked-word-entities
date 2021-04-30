### IMPORTS

from nltk.stem import PorterStemmer
from lama.modules.bert_connector import *
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics
import jsonlines
from tqdm import tqdm

### FUNCTIONS

def task11_label(obj,experiment_result):

    enity = obj['entity']['mention'].lower()
    entity_pred = experiment_result['topk'][0]['token_word_form'].lower()

    if enity != entity_pred:
        pred_label = 'REFUTES'
        if obj['label'] == pred_label:
            acc_val = 1
        else:
            acc_val = 0

    elif enity == entity_pred:
        pred_label = 'SUPPORTS'
        if obj['label'] == pred_label:
            acc_val = 1
        else:
            acc_val = 0
    return acc_val

#### MAIN ###
class BertArgs:
  def __init__(self):
    self.bert_model_dir = "/content/drive/My Drive/DMT_HW3/bert/cased_L-12_H-768_A-12"
    self.bert_model_name = "bert-base-cased"
    self.bert_vocab_name = "vocab.txt"

args = BertArgs()
model = Bert(args)

dev_set_file = '/content/drive/My Drive/DMT_HW3/dev_fever.json'
# lines counter
tot_lines = 0
# sum of accuracy for all the lines in dataset
tot_line_acc = 0

# count accuracy for dev set
with jsonlines.open(dev_set_file) as reader:
  for obj in tqdm(reader):
    tot_lines = tot_lines + 1 # lines counter
    # mask token in the middle of sentence 
    text = [obj['claim'][:obj['entity']['start_character']] + '[MASK]' + obj['claim'][obj['entity']['end_character']:]]
    # from lama/eval_generation.py
    original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([text], try_cuda=True)
    index_list = None
    filtered_log_probs_list = original_log_probs_list
    # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
    if masked_indices and len(masked_indices) > 0:
        MRR, P_AT_X, experiment_result, return_msg = evaluation_metrics.get_ranking(filtered_log_probs_list[0],\
             masked_indices, model.vocab, index_list=index_list,print_generation=False)
    
    # compute accuracy
    line_acc = task11_label(obj,experiment_result) # accuracy of each line
    tot_line_acc = tot_line_acc + line_acc

print('Task1.1  accuracy: ', tot_line_acc / tot_lines)


# result: Task1.1  accuracy:  0.549792531120332


