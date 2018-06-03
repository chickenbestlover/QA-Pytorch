# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Modification:
#   - add 'pos' and 'ner' features.
#   - use gradient hook (instead of tensor copying) for gradient masking
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from drqa.cove.cove import MTLSTM
from drqa.cove.layers import StackedLSTM, Dropout, FullAttention, WordAttention, Summ, PointerNet


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt
        self.device = 'cuda' if opt['cuda'] else 'cpu'

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                self.embedding.weight.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                offset = self.opt['tune_partial'] + 2

                def embedding_hook(grad, offset=offset):
                    grad[offset:] = 0
                    return grad

                self.embedding.weight.register_hook(embedding_hook)

        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)

        # Input size to RNN: word emb + question emb + manual features
        opt['cove_dim']=600
        doc_input_size = 2 * opt['embedding_dim'] + opt['num_features'] + opt['cove_dim']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']


        question_input_size = opt['embedding_dim'] + opt['cove_dim']

        self.x1_ratio1 = torch.nn.Parameter(0.5 * torch.ones(1))
        self.x2_ratio1 = torch.nn.Parameter(0.5 * torch.ones(1))

        self.cove_rnn = MTLSTM(opt,
                               embedding=embedding)
        self.word_attention_layer = WordAttention(input_size = opt['embedding_dim'],
                                                  hidden_size = opt['attention_size'],
                                                  dropout = opt['dropout_emb'],
                                                  device= self.device)

        self.low_doc_rnn = StackedLSTM(input_size=doc_input_size,
                                       hidden_size=opt['hidden_size'],
                                       num_layers=1,
                                       dropout=opt['dropout_emb'],
                                       device= self.device)

        self.low_ques_rnn = StackedLSTM(input_size=question_input_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=1,
                                        dropout=opt['dropout_emb'],
                                        device= self.device)

        # Output sizes of low rnn encoders
        high_doc_hidden_size = 2 * opt['hidden_size']
        high_ques_hidden_size = 2 * opt['hidden_size']

        self.high_doc_rnn = StackedLSTM(input_size=high_doc_hidden_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=1,
                                        dropout=opt['dropout_emb'],
                                        device= self.device)

        self.high_ques_rnn = StackedLSTM(input_size=high_ques_hidden_size,
                                         hidden_size=opt['hidden_size'],
                                         num_layers=1,
                                         dropout=opt['dropout_emb'],
                                         device= self.device)

        und_q_word_size = 2 * (2 * opt['hidden_size'])

        self.und_ques_rnn = StackedLSTM(input_size=und_q_word_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=1,
                                        dropout=opt['dropout_emb'],
                                        device= self.device)

        attention_inp_size = opt['embedding_dim'] + opt['cove_dim']+ 2 * (2 * opt['hidden_size'])

        self.low_attention_layer = FullAttention(input_size = attention_inp_size,
                                                 hidden_size = opt['attention_size'],
                                                 dropout = opt['dropout_emb'],
                                                 device= self.device)

        self.high_attention_layer = FullAttention(input_size=attention_inp_size,
                                                  hidden_size=opt['attention_size'],
                                                  dropout=opt['dropout_emb'],
                                                  device=self.device)

        self.und_attention_layer = FullAttention(input_size=attention_inp_size,
                                                 hidden_size=opt['attention_size'],
                                                 dropout=opt['dropout_emb'],
                                                 device=self.device)

        fuse_inp_size = 5 * (2 * opt['hidden_size'])

        self.fuse_rnn = StackedLSTM(input_size = fuse_inp_size,
                                    hidden_size = opt['hidden_size'],
                                    num_layers = 1,
                                    dropout = opt['dropout_emb'],
                                    device=self.device)

        self_attention_inp_size = opt['embedding_dim'] + \
                                  opt['cove_dim'] + \
                                  opt['pos_size'] + opt['ner_size'] + opt['num_features'] +\
                                  6 * (2 * opt['hidden_size'])# + 1


        self.self_attention_layer = FullAttention(input_size=self_attention_inp_size,
                                                  hidden_size=opt['attention_size'],
                                                  dropout=opt['dropout_emb'],
                                                  device=self.device)

        self.self_rnn = StackedLSTM(input_size = 2 * (2 * opt['hidden_size']),
                                    hidden_size = opt['hidden_size'],
                                    num_layers = 1,
                                    dropout = opt['dropout_emb'],
                                    device=self.device)

        self.summ_layer = Summ(input_size=2 * opt['hidden_size'],
                               dropout=opt['dropout_emb'],
                               device=self.device)

        self.pointer_layer = PointerNet(input_size=2 * opt['hidden_size'],
                                        dropout=opt['dropout_emb'],
                                        device=self.device)



    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask,x1_elmo=None,x2_elmo=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        ### Glove ###
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        ### cove ###
        _, x1_cove = self.cove_rnn(x1, x1_mask)
        _, x2_cove = self.cove_rnn(x2, x2_mask)

        ### embedding dropout ###
        x1_emb = Dropout(x1_emb, self.opt['dropout_emb'], self.training, device = self.device)
        x2_emb = Dropout(x2_emb, self.opt['dropout_emb'], self.training, device = self.device)
        x1_cove = Dropout(x1_cove, self.opt['dropout_emb'], self.training, device = self.device)
        x2_cove = Dropout(x2_cove, self.opt['dropout_emb'], self.training, device = self.device)
        if self.opt['pos']:
            if self.opt['dropout_emb'] > 0:
                x1_pos = nn.functional.dropout(x1_pos, p=self.opt['dropout_emb'],
                                               training=self.training)
        if self.opt['ner']:
            if self.opt['dropout_emb'] > 0:
                x1_ner = nn.functional.dropout(x1_ner, p=self.opt['dropout_emb'],
                                               training=self.training)

        word_attention_outputs = self.word_attention_layer(x1_emb, x1_mask, x2_emb, x2_mask, self.training)
        x1_word_input = torch.cat(
            [x1_emb, x1_cove, x1_pos, x1_ner, x1_f, word_attention_outputs], dim=2)
        x2_word_input = torch.cat([x2_emb, x2_cove], dim=2)

        ### low, high, understanding encoding ###
        low_x1_states = self.low_doc_rnn(x1_word_input, self.training)
        low_x2_states = self.low_ques_rnn(x2_word_input, self.training)

        high_x1_states = self.high_doc_rnn(low_x1_states, self.training)
        high_x2_states = self.high_ques_rnn(low_x2_states, self.training)

        und_x2_input = torch.cat([low_x2_states, high_x2_states], dim=2)
        und_x2_states = self.und_ques_rnn(und_x2_input, self.training)

        ########################################################################################


        ### Full Attention ###

        x1_How = torch.cat([x1_emb, x1_cove, low_x1_states, high_x1_states], dim=2)
        x2_How = torch.cat([x2_emb, x2_cove, low_x2_states, high_x2_states], dim=2)

        low_attention_outputs = self.low_attention_layer(x1_How, x1_mask, x2_How, x2_mask, low_x2_states, self.training)
        high_attention_outputs = self.high_attention_layer(x1_How, x1_mask, x2_How, x2_mask, high_x2_states, self.training)
        und_attention_outputs = self.und_attention_layer(x1_How, x1_mask, x2_How, x2_mask, und_x2_states, self.training)

        fuse_inp = torch.cat([low_x1_states, high_x1_states, low_attention_outputs, high_attention_outputs, und_attention_outputs], dim = 2)

        fused_x1_states = self.fuse_rnn(fuse_inp, self.training)

        ### Self Full Attention ###

        x1_How = torch.cat([x1_emb, x1_cove, x1_pos, x1_ner, x1_f,
                            low_x1_states, high_x1_states,
                            low_attention_outputs, high_attention_outputs,
                            und_attention_outputs, fused_x1_states], dim=2)

        self_attention_outputs = self.self_attention_layer(x1_How, x1_mask, x1_How, x1_mask, fused_x1_states, self.training)

        self_inp = torch.cat([fused_x1_states, self_attention_outputs], dim=2)

        und_doc_states = self.self_rnn(self_inp, self.training)

        ### ques summ vector ###
        init_states = self.summ_layer(und_x2_states, x2_mask, self.training)

        ### Pointer Network ###
        logits1, logits2 = self.pointer_layer.forward(und_doc_states, x1_mask, init_states, self.training)
        if self.training:
            # In training we output log-softmax for NLL
            logits1 = F.log_softmax(logits1,dim=1)
            logits2 = F.log_softmax(logits2,dim=1)
        else:
            # ...Otherwise 0-1 probabilities
            logits1 = F.softmax(logits1,dim=1)
            logits2 = F.softmax(logits2, dim=1)

        return logits1, logits2


