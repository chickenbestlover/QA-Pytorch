# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import ujson as json
from torch.autograd import Variable
from .utils import AverageMeter
from .network import ReaderNet
from qa.evaluation import evaluate
#from .rnn_reader2 import RnnDocReader
# Modification:
#   - change the logger name
#   - save & load "state_dict"s of optimizer and loss meter
#   - save all random seeds
#   - change the dimension of inputs (for POS and NER features)
#   - remove "reset parameters" and use a gradient hook for gradient masking
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

logger = logging.getLogger(__name__)

def masked_cross_entropy(logits, target, target_mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        target_mask: A Variable containing a ByteTensor of size (batch, max_len)
            which contains the mask of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    target_length = target.size(1)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    #print('logits_flat_grad', logits_flat.requires_grad)

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat,dim=1)
    #print('log_probs_grad', log_probs_flat.requires_grad)

    # target_flat: (batch * max_len, 1)
    target_flat = target.contiguous().view(-1, 1)
    #print('target_flat_grad', target.requires_grad)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    #print('losses_flat_grad', losses_flat.requires_grad)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)

    losses.data.masked_fill_(target_mask.data.byte(),0)
    #print('losses_grad', losses.requires_grad)
    #print(target_mask)
    loss = losses.sum() / torch.nonzero(losses.data).size(0)
    #print('loss_grad',loss.requires_grad)
    return loss

class QAModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.device = torch.cuda.current_device() if opt['cuda'] else torch.device('cpu')
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        self.train_loss_q_autodecoded = AverageMeter()
        self.train_loss_q_decoded = AverageMeter()
        if state_dict:
            self.train_loss.load(state_dict['loss'])
            self.train_loss_q_autodecoded.load(state_dict['loss_q_auto'])
            self.train_loss_q_decoded.load(state_dict['loss_q_decoded'])
        # Building network.
        self.network = ReaderNet(opt, embedding=embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
            self.network.embedding.weight.data = state_dict['network']['embedding.weight'].data
        self.network.to(self.device)

        # Building optimizer.
        self.opt_state_dict = state_dict['optimizer'] if state_dict else None
        self.build_optimizer()

    def build_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.opt['learning_rate'],
                                       momentum=self.opt['momentum'],
                                       weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,self.opt['learning_rate'],
                                          weight_decay=self.opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.opt['optimizer'])
        if self.opt_state_dict:
            self.optimizer.load_state_dict(self.opt_state_dict)

        num_params = sum(p.data.numel() for p in parameters
                         if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr())
        print("{} parameters".format(num_params),'\n')


    def update(self, ex):
        with torch.enable_grad():
            # Train mode
            self.network.train()

            # Transfer to GPU
            target_s = ex[-3].to(self.device)
            target_e = ex[-2].to(self.device)

            # Run forward
            score_s, score_e, ques_autodecoded, ques_decoded = self.network(*ex[:-1])

            # Compute loss and accuracies
            loss1 = F.cross_entropy(score_s, target_s) + F.cross_entropy(score_e, target_e)
            loss2 = masked_cross_entropy(ques_autodecoded,ex[9], ex[13])
            loss3 = masked_cross_entropy(ques_decoded, ex[9], ex[13])

            loss = loss1 + loss2 + loss3
            self.train_loss.update(loss1.item())
            self.train_loss_q_autodecoded.update(loss2.item())
            self.train_loss_q_decoded.update(loss3.item())
            # Clear gradients and run backward
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                          self.opt['grad_clipping'])

            # Update parameters
            self.optimizer.step()
            self.updates += 1

            self.network.reset_parameters()


    def get_predictions(self, logits1, logits2, maxlen=15) :
        batch_size, P = logits1.size()
        outer = torch.matmul(F.softmax(logits1, -1).unsqueeze(2),
                             F.softmax(logits2, -1).unsqueeze(1))

        band_mask = Variable(torch.zeros(P, P)).to(self.device)

        for i in range(P) :
            band_mask[i, i:max(i+maxlen, P)].data.fill_(1.0)

        band_mask = band_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        outer = outer * band_mask

        yp1 = torch.max(torch.max(outer, 2)[0], 1)[1]
        yp2 = torch.max(torch.max(outer, 1)[0], 1)[1]

        return yp1, yp2


    def convert_tokens(self, eval_file, qa_id, pp1, pp2) :
        answer_dict = {}
        remapped_dict = {}
        for qid, p1, p2 in zip(qa_id, pp1, pp2) :

            p1 = int(p1)
            p2 = int(p2)
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            uuid = eval_file[str(qid)]["uuid"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx : end_idx]
            remapped_dict[uuid] = context[start_idx : end_idx]
        return answer_dict, remapped_dict

    def Evaluate(self, batches, eval_file=None, answer_file = None) :
        with open(eval_file, 'r') as f :
            #print('Start evaluate...')
            eval_file = json.load(f)
            answer_dict = {}
            #remapped_dict = {}
            with torch.no_grad():
                self.network.eval()
                for i,batch in enumerate(batches):
                    start_score,end_score = self.network.forward(*batch[:-3])
                    y1, y2 = self.get_predictions(start_score,end_score )
                    qa_id = batch[-1]
                    answer_dict_, remapped_dict_ = self.convert_tokens(eval_file, qa_id, y1, y2)
                    answer_dict.update(answer_dict_)
                    #remapped_dict.update(remapped_dict_)
                    del y1, y2, answer_dict_#, remapped_dict_
                    #print('> evaluating [{}/{}]'.format(i, len(batches)))
            metrics = evaluate(eval_file, answer_dict)
            #with open(answer_file, 'w') as f:
            #    json.dump(remapped_dict, f)

        return metrics['exact_match'], metrics['f1']

    def save(self, filename, epoch, scores):
        em, f1, best_eval = scores
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates,
                'loss': self.train_loss.state_dict(),
                'loss_q_auto': self.train_loss_q_autodecoded.state_dict(),
                'loss_q_decoded': self.train_loss_q_decoded.state_dict(),
            },
            'config': self.opt,
            'epoch': epoch,
            'em': em,
            'f1': f1,
            'best_eval': best_eval,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')



