import re
import os
import sys
import math
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
from drqa.model_ELMo_fusion import DocReaderModel
from drqa.utils import str2bool
from allennlp.modules.elmo import Elmo, batch_to_ids
import os
import ujson as json
import numpy as np


def main():

    if not os.path.exists('train_model/') :
        os.makedirs('train_model/')
    if not os.path.exists('result/') :
        os.makedirs('result/')

    args, log = setup()
    log.info('[Program starts. Loading data...]')
    #train, dev, dev_y, embedding, opt = load_data(vars(args))
    train, dev, word2id, char2id, embedding, opt = load_data(vars(args))
    log.info(opt)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(args.model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)


        epoch_0 = checkpoint['epoch'] + 1
        # synchronize random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        #if args.reduce_lr:
        #    lr_decay(model.optimizer, lr_decay=args.reduce_lr)
        #    log.info('[learning rate reduced by {}]'.format(args.reduce_lr))
        batches = BatchGen(dev, batch_size=args.batch_size)
        em,f1= model.Evaluate(batches, args.data_path + 'dev_eval.json',
                       answer_file='result/' + args.model_dir.split('/')[-1] + '.answers')
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        if math.fabs(em - checkpoint['em']) > 1e-3 or math.fabs(f1 - checkpoint['f1']) > 1e-3:
            log.info('Inconsistent: recorded EM: {} F1: {}'.format(checkpoint['em'], checkpoint['f1']))
            log.error('Error loading model: current code is inconsistent with code used to train the previous model.')
            #exit(1)
        best_val_score = checkpoint['best_eval']
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1
        best_val_score = 0.0




    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, batch_size=args.batch_size, device='cuda' if args.cuda else 'cpu')
        start = datetime.now()

        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.value,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))

        log.debug('\n')

        # # eval
        batches = BatchGen(dev, batch_size=args.batch_size, device='cuda' if args.cuda else 'cpu')
        em, f1 = model.Evaluate(batches, args.data_path + 'dev_eval.json',
                                answer_file='result/' + args.model_dir.split('/')[-1] + '.answers')
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        # save
        model_file = os.path.join(args.model_dir, 'checkpoint_elmo_fusion.pt'.format(epoch))
        model.save(model_file, epoch, [em, f1, best_val_score])
        if f1 > best_val_score:
            best_val_score = f1
            copyfile(
                model_file,
                os.path.join(args.model_dir, 'best_model_elmo_fusion.pt'))
            log.info('[new best model saved.]')

        if epoch > 0 and epoch % args.decay_period == 0:
            model.optimizer = lr_decay(model.optimizer,lr_decay= args.reduce_lr)



def setup():
    parser = argparse.ArgumentParser(
        description='Train a Document Reader model.'
    )
    # system

    parser.add_argument('--data_path', default='./SQuAD_fusion/')
    parser.add_argument('--logfile', type=str, default='log_fusion.txt',
                        help='logfile name')
    parser.add_argument('--log_per_updates', type=int, default=3,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('-e', '--epochs', type=int, default=150)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-rs', '--resume', default='best_model_elmo_fusion.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('--decay_period', type=int, default=15)
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.5,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002,
                        help='applied to SGD and Adamax.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    # model

    parser.add_argument('--use_char', type=bool, default=False)
    parser.add_argument('--MTLSTM_path', type=str, default='./drqa/cove/MT-LSTM.pth')
    parser.add_argument('--char_hidden_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--attention_size', type=int, default=250)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--char_dim', type=int, default=50)
    parser.add_argument('--pos_dim', type=int, default=12)
    parser.add_argument('--ner_dim', type=int, default=8)
    parser.add_argument('--dropout_emb', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_len', type=int, default=15)
    args = parser.parse_args()

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model_elmo_fusion.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        # means we're starting fresh
        args.resume = ''

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # setup logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
            else:
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, args.logfile))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate=False
    return args, log


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

#
# def load_data(opt):
#     with open('SQuAD/meta.msgpack', 'rb') as f:
#         meta = msgpack.load(f, encoding='utf8')
#     embedding = torch.Tensor(meta['embedding'])
#     opt['pretrained_words'] = True
#     opt['vocab_size'] = embedding.size(0)
#     opt['embedding_dim'] = embedding.size(1)
#     opt['pos_size'] = len(meta['vocab_tag'])
#     opt['ner_size'] = len(meta['vocab_ent'])
#     BatchGen.pos_size = opt['pos_size']
#     BatchGen.ner_size = opt['ner_size']
#     with open(opt['data_file'], 'rb') as f:
#         data = msgpack.load(f, encoding='utf8')
#     train = data['train']
#     data['dev'].sort(key=lambda x: len(x[1]))
#     dev = [x[:8] + x[9:]for x in data['dev']]
#     dev_y = [x[8] for x in data['dev']]
#     return train, dev, dev_y, embedding, opt


def get_data(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = json.load(f)
    return data

def load_data(opt):
    print('load data...')
    data_path = opt['data_path']

    train_data = get_data(data_path + 'train.json')
    dev_data = get_data(data_path + 'dev.json')
    word2id = get_data(data_path + 'word2id.json')
    char2id = get_data(data_path + 'char2id.json')
    pos2id = get_data(data_path + 'pos2id.json')
    ner2id = get_data(data_path + 'ner2id.json')

    opt['char_size'] = int(np.max(list(char2id.values())) + 1)
    opt['pos_size'] = int(np.max(list(pos2id.values())) + 1)
    opt['ner_size'] = int(np.max(list(ner2id.values())) + 1)

    print('load embedding...')
    word_emb = np.array(get_data(data_path + 'word_emb.json'), dtype=np.float32)
    embedding = torch.from_numpy(word_emb)
    del word_emb

    return train_data, dev_data, word2id, char2id, embedding, opt



class BatchGen:
    def __init__(self, data, batch_size, device='cuda'):
        """
        input:
            data - list of lists
            batch_size - int
        """
        #options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        #weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        #self.elmo = Elmo(options_file, weight_file, 2, dropout=0).to('cuda')
        self.batch_size = batch_size
        self.data = data
        self.device = device

    def compute_mask(self, x):
        return torch.eq(x, 0).to(self.device)

    def __len__(self):
        return len(self.data['context_ids'])//self.batch_size +1

    def __iter__(self):
        for i in range(0, len(self.data['context_ids']), self.batch_size):
            #context_elmo = self.elmo(context_elmo)['elmo_representations']
            #question_elmo = self.elmo(question_elmo)['elmo_representations']
            batch_data = (self.data['context_ids'][i:i + self.batch_size],
                          self.data['context_char_ids'][i:i + self.batch_size],
                          self.data['context_pos_ids'][i:i + self.batch_size],
                          self.data['context_ner_ids'][i:i + self.batch_size],
                          self.data['context_match_origin'][i:i + self.batch_size],
                          self.data['context_match_lower'][i:i + self.batch_size],
                          self.data['context_match_lemma'][i:i + self.batch_size],
                          self.data['context_tf'][i:i + self.batch_size],
                          self.data['ques_ids'][i:i + self.batch_size],
                          self.data['ques_char_ids'][i:i + self.batch_size],
                          self.data['ques_pos_ids'][i:i + self.batch_size],
                          self.data['ques_ner_ids'][i:i + self.batch_size],
                          self.data['y1'][i:i + self.batch_size],
                          self.data['y2'][i:i + self.batch_size],
                          self.data['id'][i:i + self.batch_size])

            """
            batch_data[0] : passage_ids,
            batch_data[1] : passage_char_ids,
            batch_data[2] : passage_pos_ids,
            batch_data[3] : passage_ner_ids,
            batch_data[4] : passage_match_origin,
            batch_data[5] : passage_match_lower,
            batch_data[6] : passage_match_lemma,
            batch_data[7] : passage_tf,
            batch_data[8] : ques_ids,
            batch_data[9] : ques_char_ids,
            batch_data[10] : ques_pos_ids,
            batch_data[11] : ques_ner_ids,
            batch_data[12] : y1,
            batch_data[13] : y2,
            batch_data[14] : id
            """

            passage_ids = torch.LongTensor(batch_data[0]).to(self.device)
            passage_char_ids = torch.LongTensor(batch_data[1]).to(self.device)
            passage_pos_ids = torch.LongTensor(batch_data[2]).to(self.device)
            passage_ner_ids = torch.LongTensor(batch_data[3]).to(self.device)
            passage_match_origin = torch.FloatTensor(batch_data[4]).to(self.device)
            passage_match_lower = torch.FloatTensor(batch_data[5]).to(self.device)
            passage_match_lemma = torch.FloatTensor(batch_data[6]).to(self.device)
            passage_tf = torch.FloatTensor(batch_data[7]).to(self.device)


            ques_ids = torch.LongTensor(batch_data[8]).to(self.device)
            ques_char_ids = torch.LongTensor(batch_data[9]).to(self.device)
            ques_pos_ids = torch.LongTensor(batch_data[10]).to(self.device)
            ques_ner_ids = torch.LongTensor(batch_data[11]).to(self.device)


            y1 = torch.LongTensor(batch_data[12]).to(self.device)
            y2 = torch.LongTensor(batch_data[13]).to(self.device)

            id=batch_data[14]

            del batch_data

            p_lengths = passage_ids.ne(0).long().sum(1)
            q_lengths = ques_ids.ne(0).long().sum(1)

            passage_maxlen = int(torch.max(p_lengths, 0)[0])
            ques_maxlen = int(torch.max(q_lengths, 0)[0])

            passage_ids = passage_ids[:, :passage_maxlen]
            passage_char_ids = passage_char_ids[:, :passage_maxlen]
            passage_pos_ids = passage_pos_ids[:, :passage_maxlen]
            passage_ner_ids = passage_ner_ids[:, :passage_maxlen]
            passage_match_origin = passage_match_origin[:, :passage_maxlen]
            passage_match_lower = passage_match_lower[:, :passage_maxlen]
            passage_match_lemma = passage_match_lemma[:, :passage_maxlen]
            passage_tf = passage_tf[:, :passage_maxlen]
            ques_ids = ques_ids[:, :ques_maxlen]
            ques_char_ids = ques_char_ids[:, :ques_maxlen]
            ques_pos_ids = ques_pos_ids[:, :ques_maxlen]
            ques_ner_ids = ques_ner_ids[:, :ques_maxlen]

            p_mask = self.compute_mask(passage_ids)
            q_mask = self.compute_mask(ques_ids)


            yield (passage_ids,
                   passage_char_ids,
                   passage_pos_ids,
                   passage_ner_ids,
                   passage_match_origin.unsqueeze(2).float(),
                   passage_match_lower.unsqueeze(2).float(),
                   passage_match_lemma.unsqueeze(2).float(),
                   passage_tf.unsqueeze(2),
                   p_mask,
                   ques_ids,
                   ques_char_ids,
                   ques_pos_ids,
                   ques_ner_ids,
                   q_mask,
                   y1,
                   y2,
                   id)




def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


if __name__ == '__main__':
    main()

