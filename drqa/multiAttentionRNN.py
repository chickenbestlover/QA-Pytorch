# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import layers2 as layers

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

import cuda_functional as MF
import numpy as np


class AttentionRNN(nn.Module):
    def __init__(self, opt, doc_input_size, question_input_size,ratio=2):
        super(AttentionRNN, self).__init__()
        self.doc_rnns = nn.ModuleList()
        self.question_rnns = nn.ModuleList()

        self.doc_attns = nn.ModuleList()
        self.question_attns = nn.ModuleList()
        self.doc_convs=nn.ModuleList()
        self.dropout_rate =opt['dropout_rnn']
        self.question_convs=nn.ModuleList()
        self.num_layers = opt['doc_layers']
        self.ratio=ratio
        for i in range(self.num_layers):

            doc_input_size = doc_input_size if i == 0 else 2 * opt['hidden_size']+opt['hidden_size']//ratio
            question_input_size = question_input_size if i == 0 else 2 * opt['hidden_size']+opt['hidden_size']//ratio

            self.doc_rnns.append(layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=1,
            dropout_rate=opt['dropout_rnn'],
            ))
            self.question_rnns.append(layers.StackedBRNN(
            input_size=question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=1,
            dropout_rate=opt['dropout_rnn'],
            ))

            self.doc_attns.append(layers.multiSeqAttnMatch(input_size=2 * opt['hidden_size'],n_head=opt['num_heads']))
            self.question_attns.append(layers.multiSeqAttnMatch(input_size=2 * opt['hidden_size'],n_head=opt['num_heads']))
            self.doc_convs.append(layers.Conv1by1DimReduce(in_channels=2 * opt['hidden_size'], out_channels=opt['hidden_size'] // ratio))

            self.question_convs.append(layers.Conv1by1DimReduce(in_channels=2 * opt['hidden_size'], out_channels=opt['hidden_size'] // ratio))

    def forward(self, x1, x1_mask, x2, x2_mask):

        # Encode all layers

        for i in range(self.num_layers):
            # Forward
            #print('doc_rnn_input:',doc_rnn_input.size())
            #print(i,' x1',x1.size())
            #print(i,'x1_mask',x1_mask.size())
            x1 = self.doc_rnns[i](x1,x1_mask)
            x1 = nn.functional.dropout(x1, p=self.dropout_rate,training=self.training)
            #print(i,' x1',x1.size())
            #print(i,' x2',x2.size())
            x2 = self.question_rnns[i](x2,x2_mask)
            x2 = nn.functional.dropout(x2, p=self.dropout_rate,training=self.training)
            #q_merge_weights = self.question_self_attns[i](x2, x2_mask)
            #question_hidden = layers.weighted_avg(x2, q_merge_weights)
            matched_x2_hiddens = self.doc_attns[i](x1,x2,x2_mask)
            #matched_x2_hiddens = nn.functional.dropout(matched_x2_hiddens, p=self.dropout_rate, training=self.training)
            matched_x2_hiddens = self.doc_convs[i](matched_x2_hiddens)
            #matched_x2_hiddens = nn.functional.dropout(matched_x2_hiddens, p=self.dropout_rate,training=self.training)
            matched_x1_hiddens = self.doc_attns[i](x2,x1,x1_mask)
            #matched_x1_hiddens = nn.functional.dropout(matched_x1_hiddens, p=self.dropout_rate,training=self.training)
            matched_x1_hiddens = self.doc_convs[i](matched_x1_hiddens)
            #matched_x1_hiddens = nn.functional.dropout(matched_x1_hiddens, p=self.dropout_rate,training=self.training)
            #print(i,' hidden:',matched_x2_hiddens.size())
            #print(i,' x1:',x1.size())
            x1 = torch.cat([x1,matched_x2_hiddens],dim=2)
            x2 = torch.cat([x2, matched_x1_hiddens], dim=2)

        return x1.contiguous(), x2.contiguous()

