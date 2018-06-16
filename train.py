import sys
import math
import random
import logging
import argparse
from shutil import copyfile
from datetime import datetime
import torch
from qa.model import QAModel
from qa.utils import str2bool
from allennlp.modules.elmo import batch_to_ids
import os
import ujson as json
import numpy as np

def setup():
    parser = argparse.ArgumentParser(
        description='Train a QA model.'
    )
    # system

    parser.add_argument('--data_path', default='./SQuAD/')
    parser.add_argument('--logfile', type=str, default='log.txt',
                        help='logfile name')
    parser.add_argument('--log_per_updates', type=int, default=3,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('-e', '--epochs', type=int, default=150)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-rs', '--resume', default='best_model.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('--decay_period', type=int, default=20)
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
    # model
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--rnn_type', type=str, default='lstm')
    parser.add_argument('--use_char', action='store_true')
    parser.add_argument('--use_cove', action='store_true')
    parser.add_argument('--use_elmo', action='store_true')
    parser.add_argument('--use_res', action='store_true')
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--MTLSTM_path', type=str, default='./qa/cove/MT-LSTM.pth')
    parser.add_argument('--char_hidden_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--attention_size', type=int, default=250)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--char_dim', type=int, default=50)
    parser.add_argument('--pos_dim', type=int, default=12)
    parser.add_argument('--ner_dim', type=int, default=8)
    parser.add_argument('--dropout_emb', type=float, default=0.45)
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--dropout_rnn',type=float,default=0)
    parser.add_argument('--max_len', type=int, default=15)
    args = parser.parse_args()

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
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

def get_data(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = json.load(f)
    return data

def load_data(opt):
    print('load data...')
    data_path = opt['data_path']

    train_data = get_data(data_path + 'train_elmo.json')
    dev_data = get_data(data_path + 'dev_elmo.json')
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
    def __init__(self,opt, data, batch_size, device='cuda'):
        """
        input:
            data - list of lists
            batch_size - int
        """

        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.opt = opt

    def compute_mask(self, x):
        return torch.eq(x, 0).to(self.device)

    def __len__(self):
        return len(self.data['context_ids'])//self.batch_size +1

    def __iter__(self):
        for i in range(0, len(self.data['context_ids']), self.batch_size):
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
                          self.data['id'][i:i + self.batch_size],
                          self.data['context_tokens'][i:i + self.batch_size],
                          self.data['ques_tokens'][i:i + self.batch_size])

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
            batch_data[14] : id,
            batch_data[15] : passage_tokens,
            batch_data[16] : ques_tokens
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

            if self.opt['use_elmo']:
                passage_elmo_ids = batch_to_ids(batch_data[15]).to(self.device)
                question_elmo_ids = batch_to_ids(batch_data[16]).to(self.device)
            else:
                passage_elmo_ids = None
                question_elmo_ids = None

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
                   passage_elmo_ids,
                   question_elmo_ids,
                   y1,
                   y2,
                   id)

def param_equility_check(model1,model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print(p1.size())
    return True


if not os.path.exists('result/') :
    os.makedirs('result/')

args, log = setup()
log.info('[Program starts. Loading data...]')
train, dev, word2id, char2id, embedding, opt = load_data(vars(args))
del word2id, char2id
log.info(opt)
log.info('[Data loaded.]')

if args.resume:
    log.info('[loading previous model...]')
    checkpoint = torch.load(os.path.join(args.model_dir, args.resume))
    if args.resume_options:
        opt = checkpoint['config']
    state_dict = checkpoint['state_dict']
    model = QAModel(opt, embedding, state_dict)
    epoch_0 = checkpoint['epoch'] + 1
    # synchronize random seed
    random.setstate(checkpoint['random_state'])
    torch.random.set_rng_state(checkpoint['torch_state'])
    if args.cuda:
        torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
    batches = BatchGen(opt, dev, batch_size=args.batch_size)
    em,f1= model.Evaluate(batches, args.data_path + 'dev_eval.json',
                   answer_file='result/' + args.model_dir.split('/')[-1] + '.answers')
    log.info("[dev EM: {} F1: {}]".format(em, f1))
    if math.fabs(em - checkpoint['em']) > 1e-3 or math.fabs(f1 - checkpoint['f1']) > 1e-3:
        log.info('Inconsistent: recorded EM: {} F1: {}'.format(checkpoint['em'], checkpoint['f1']))
        log.error('Error loading model: current code is inconsistent with code used to train the previous model.')
        #exit(1)
    best_val_score = checkpoint['best_eval']
    del batches, state_dict, checkpoint, embedding
else:
    model = QAModel(opt, embedding)
    del embedding
    epoch_0 = 1
    best_val_score = 0.0




for epoch in range(epoch_0, epoch_0 + args.epochs):

    log.warning('Epoch {}'.format(epoch))

    # train
    batches = BatchGen(opt, train, batch_size=args.batch_size, device='cuda' if args.cuda else 'cpu')
    start = datetime.now()
    for i, batch in enumerate(batches):
        model.update(batch)
        if i % args.log_per_updates == 0:
            log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                epoch, model.updates, model.train_loss.value,
                str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
    log.debug('\n')
    del batches

    # # eval
    batches = BatchGen(opt, dev, batch_size=args.batch_size, device='cuda' if args.cuda else 'cpu')
    em, f1 = model.Evaluate(batches, args.data_path + 'dev_eval.json',
                            answer_file='result/' + args.model_dir.split('/')[-1] + '.answers')
    log.info("[dev EM: {} F1: {}] \n".format(em, f1))
    #log.debug('\n')
    del batches

    # save
    model_file = os.path.join(args.model_dir,
                              'checkpoint'+
                              '_char'+str(opt['use_char'])+
                              '_cove'+str(opt['use_cove'])+
                              '_elmo'+str(opt['use_elmo'])+
                              '_hidden'+str(opt['hidden_size'])+
                              '_' + str(opt['rnn_type']) + str(opt['num_layers'])+
                              '_res' + str(opt['use_res']) +
                              '.pt')
    model.save(model_file, epoch, [em, f1, best_val_score])
    if f1 > best_val_score:
        best_val_score = f1
        copyfile(
            model_file,
            os.path.join(args.model_dir,
                         'char'+str(opt['use_char'])+
                         '_cove'+str(opt['use_cove'])+
                         '_elmo'+str(opt['use_elmo'])+
                         '_hidden' + str(opt['hidden_size']) +
                         '_' + str(opt['rnn_type']) + str(opt['num_layers']) +
                         '_res' + str(opt['use_res']) +
                         '.pt'))

        log.info('[new best model saved.] \n')
    if epoch > 0 and epoch % args.decay_period == 0:
        model.optimizer = lr_decay(model.optimizer,lr_decay= args.reduce_lr)
        log.info('> learning rate decayed by 0.5 \n')
