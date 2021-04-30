### IMPORTS

from lama.modules.bert_connector import *
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics
import jsonlines
from tqdm import tqdm
import matplotlib.pyplot as plt

# FUNCTIONS

def task13_label(obj,experiment_result,th_lst):
    ''' function appends 1 if there is true prediction with probability > threshold
    if probability of true prediction is smaller then 0, same if there is no true prediction
    '''
    # list of probabilities, sorted from big to small
    prob_value_lst = [np.exp(experiment_result['topk'][i]['log_prob']) for i in range(len(experiment_result['topk']))]
    # list of predicted tokens
    pred_list = [experiment_result['topk'][i]['token_word_form'] for i in range(len(experiment_result['topk']))]
    # list to save 1 or 0 for each threshold

    acc_lst = []
    for t in th_lst:
        for j in range(len(prob_value_lst)):
            if prob_value_lst[j] >= t and pred_list[j] == obj['entity']['mention']:
                acc_lst.append(1)
                break
            elif prob_value_lst[j] >= t and pred_list[j] != obj['entity']['mention']:
                if j == len(prob_value_lst)-1:
                    acc_lst.append(0)
                continue          
            elif prob_value_lst[j] < t:
                acc_lst.append(0)
                break
    return acc_lst

### MAIN
class BertArgs:
  def __init__(self):
    self.bert_model_dir = "/content/drive/My Drive/DMT_HW3/bert/cased_L-12_H-768_A-12"
    self.bert_model_name = "bert-base-cased"
    self.bert_vocab_name = "vocab.txt"

args = BertArgs()
model = Bert(args)

# list which saves accuracy for each line for each threshold
tot_line_acc = []
# list of thresholds
th_lst = np.linspace(1, 0, num=350)
dev_set_file = '/content/drive/My Drive/DMT_HW3/dev_fever.json'

with jsonlines.open(dev_set_file) as reader:
  for obj in tqdm(reader):
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
    line_acc = task13_label(obj,experiment_result,th_lst) # accuracy of each line
    tot_line_acc.append(line_acc) # accuracy for all lines

dataset_true_for_th = list(map(sum, zip(*tot_line_acc))) # sum along the columns
acc_at_th = [x / len(tot_line_acc) for x in dataset_true_for_th] # acc of each threshold
# plot k vs accuracy
plt.plot(th_lst, acc_at_th)
plt.xlabel('threshold')
plt.ylabel('accuracy')
plt.show()