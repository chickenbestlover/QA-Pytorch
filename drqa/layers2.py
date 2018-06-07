# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

from cuda_functional import SRUCell
import numpy as np

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        #self.lns = nn.ModuleList()
        for i in range(num_layers):
            self.input_size = input_size if i == 0 else 2 * hidden_size+hidden_size//2
            #self.rnns.append(rnn_type(input_size, hidden_size,
            #                          num_layers=1,
            #                          bidirectional=True))
            self.rnns.append(SRUCell(input_size, hidden_size,
                                     dropout=dropout_rate,  # dropout applied between RNN layers
                                     rnn_dropout=dropout_rate,  # variational dropout applied on linear transformation
                                     use_tanh=1,  # use tanh?
                                     use_relu=0,  # use ReLU?
                                     use_selu=0,  # use SeLU?
                                     bidirectional=True,  # bidirectional RNN ?
                                     weight_norm=False,  # apply weight normalization on parameters
                                     layer_norm=False,  # apply layer normalization on the output of each layer
                                     highway_bias=0,  # initial bias of highway gate (<= 0)
                                     rescale=False
                                     ))

            #self.lns.append(LayerNorm(d_hid=2 * hidden_size))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        #if self.padding or not self.training:
        #    return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        #print(x.size())
        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
#            if self.dropout_rate > 0:
#                rnn_input = F.dropout(rnn_input,
#                                      p=self.dropout_rate,
#                                      training=self.training)
            # Forward
            #print(self.input_size)
            #print(rnn_input.size())
            rnn_output = self.rnns[i](rnn_input)[0]
            #if i > 0:
             #   rnn_output += rnn_input
             #   rnn_output = self.lns[i](rnn_output.view(-1,rnn_output.size(2))).view_as(rnn_output)

            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output.contiguous()

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)),dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq

class multiSeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size,n_head=4):
        super(multiSeqAttnMatch, self).__init__()

        self.hidden_size = input_size // n_head
        self.w = nn.Parameter(torch.FloatTensor(n_head, input_size, self.hidden_size))
        init.xavier_normal_(self.w)
        self.n_head = n_head
    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        x_s = x.repeat(self.n_head,1,1).view(self.n_head,-1,x.size(2)) # n_head * (batch x len1) * input_size
        #print('y', y.size())
        y_s = y.repeat(self.n_head, 1, 1).view(self.n_head, -1, y.size(2)) # n_head * (batch x len2) * input_size
        #print('y_s', y_s.size())
        x_s = torch.bmm(x_s,self.w) # n_head * (batch x len1) * hidden_size
        y_s = torch.bmm(y_s, self.w)  # n_head * (batch x len2) * hidden_size
        #print('y_s',y_s.size())
        x_s = x_s.view(-1,x.size(1),self.hidden_size) # (n_head x batch) * len1 * hhidden_size
        y_s = y_s.view(-1, y.size(1), self.hidden_size) # (n_head x batch) * len2 * hidden_size
        #print('y_s', y_s.size())
        #x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
        x_s_proj = F.relu(x_s)
        #y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
        y_s_proj = F.relu(y_s)

        # Compute scores
        scores = x_s_proj.bmm(y_s_proj.transpose(2, 1)) # (n_head x batch) * len1 * len2

        # Mask padding
        y_mask = y_mask.unsqueeze(1).repeat(self.n_head,x.size(1),1)
        #print('y_mask:',y_mask.size())
        #print('scores:', scores.size())

        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)),dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y_s) # (n_head x batch) * len1 * hidden_size
        matched_seq = torch.cat(torch.split(matched_seq,x.size(0),dim=0),dim=-1) # batch x len1 x (n_head * hidden_size)

        return matched_seq

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        #if self.eval(): print('Wy:', Wy.size())
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy,dim=1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy,dim=1)
        return alpha

class BilinearSeqAttn_norm(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False,activation='sigmoid'):
        super(BilinearSeqAttn_norm, self).__init__()
        self.activation=activation
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        #if self.eval(): print('Wy:', Wy.size())
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))

        alpha = F.softmax(xWy)
        #alpha = F.sigmoid(xWy)
        return alpha

class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores,dim=1)
        return alpha

class doc_LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size, output_size):
        super(doc_LinearSeqAttn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1), self.output_size)
        x_mask = x_mask.unsqueeze(2).expand_as(scores)

        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class LinearSeqAttn_ques(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn_ques, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class Conv1by1DimReduce(nn.Module):
    '''
    1by1 conv layer for feature dimension reduction
    '''
    def __init__(self,in_channels,out_channels,kernel_size=1):
        super(Conv1by1DimReduce,self).__init__()
        self.conv1by1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    def forward(self,x):
        '''
        :param x: batch * len * in_channels
        :return: batch * len * out_channels
        '''
        return torch.transpose(F.relu(self.conv1by1(torch.transpose(x, 1, 2))), 1, 2)

class convEncoder(nn.Module):
    '''
    convNet for document length reduction
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(convEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding)

        self.maxPool = torch.nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        '''
        :param x: batch * input_len * in_channels
        :return: batch * output_len * out_channels
        '''
        out= F.relu(self.maxPool(self.conv1(torch.transpose(x, 1, 2))))
        #print('out: ', out.size())

        out= F.relu(self.maxPool(self.conv2(out)))
        out= torch.transpose(F.relu(self.maxPool(self.conv3(out))), 1, 2)
        return out

class RelationNetwork(nn.Module):
    '''
    RelationNet
    '''
    def __init__(self,hidden_size, output_size):
        super(RelationNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.g_fc1 = torch.nn.Linear(hidden_size, output_size)
        #self.g_bn1 = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.ln1 = LayerNorm(d_hid=output_size)
        #self.g_fc2 = torch.nn.Linear(hidden_size, hidden_size)
        #self.g_bn2 = torch.nn.BatchNorm1d(num_features=hidden_size)
        #self.ln2 = LayerNorm(d_hid=hidden_size)

        #self.g_fc3 = torch.nn.Linear(hidden_size, hidden_size)
        #self.g_bn3 = torch.nn.BatchNorm1d(num_features=hidden_size)
        #self.ln3 = LayerNorm(d_hid=hidden_size)

        self.g_fc4 = torch.nn.Linear(output_size, output_size)
        #self.g_bn4 = torch.nn.BatchNorm1d(num_features=output_size)
        self.ln4 = LayerNorm(d_hid=output_size)

    def forward(self, doc_hiddens, question_hiddens):
        '''

        :param doc_hiddens: batch * input_len * in_channels
        :param question_hiddens: batch * input_len * in_channels
        :return: batch * output_size
        '''
        # Concatenate all available relations
        num_doc_objects = doc_hiddens.size(1)
        num_question_objects = question_hiddens.size(1)
        num_total_objects = num_doc_objects * num_question_objects
        doc = doc_hiddens.unsqueeze(1)
        doc = doc.repeat(1, num_question_objects, 1, 1)
        q = question_hiddens.unsqueeze(2).repeat(1, 1, num_doc_objects, 1)
        relations = torch.cat([doc, q], 3)
        #print('relations       :', relations.size())
        x_r = relations.view(-1,doc.size(3)+q.size(3))
        #print('x_r:',x_r.size())
        #res1 = x_r.clone()
        #print(self.g_fc1)
        x_r = F.relu((self.g_fc1(x_r)))# + res1
        x_r = self.ln1.forward(x_r)
        #res2 = x_r.clone()
        #x_r = F.relu((self.g_fc2(x_r))) + res2
        #x_r = self.ln2.forward(x_r)
        #res3 = x_r.clone()
        #x_r = F.relu((self.g_fc3(x_r))) + res3

        x_r = F.relu((self.g_fc4(x_r)))
        x_r = self.ln4.forward(x_r)

        x_g = x_r.view(relations.size(0), relations.size(1) * relations.size(2), -1)
        x_g = x_g.sum(1).squeeze(1)/num_total_objects

        return x_g

class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) \
                 + self.b_2.expand_as(ln_out)
        return ln_out

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
                                [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
                                if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)



# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1, keepdim=True).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)

def kmax_indice(x, dim, k):
    return x.topk(k, dim = dim)[1].sort(dim = dim)[0]

def indice_pooling(x, indices):
    #out = x.contiguous().transpose(1,2)
    #out = out.contiguous().view(-1,x.size(1))
    #indices = indices.unsqueeze(2).repeat(1,1,x.size(2))
    #print('indices:',indices.size())
    #print('x:',x.size())
    out=torch.cat([torch.index_select(x_, 0, i).unsqueeze(0) for x_, i in zip(x, indices)])
    #print('out:',out)
    #out = x.gather(dim=2, index=indices)
    #out = out.view(x.size(0),x.size(2),-1)
    #out = out.transpose(1,2)
    return out