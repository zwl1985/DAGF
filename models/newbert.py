import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PHMultiHeadAttention, DynamicLSTM, Attention, RelationalGraphConvLayer, GraphConvolution
from transformers import BertModel, RobertaModel


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)

    def forward(self, inputs, y_onehot={}):
        outputs1, outputs2, outputs3, kl_loss,  pooled_output = self.gcn_model(inputs)
    #   16,768      16,384   16,384                16,768
        final_outputs = torch.cat((outputs2, outputs3, pooled_output, outputs1), dim=-1)   # 16,2304
        logits = self.classifier(final_outputs)
        W1 = nn.Linear(self.opt.bert_dim, self.opt.polarities_dim).to('cuda')
        W2 = nn.Linear(self.opt.bert_dim, self.opt.polarities_dim).to('cuda')
        syn_out = outputs1
        sem_out = torch.cat((outputs2, outputs3), dim=-1)
        if y_onehot and isinstance(y_onehot, torch.Tensor):
            y_pred_syn = F.softmax(W1(syn_out), dim=-1)
            y_pred_sem = F.softmax(W2(sem_out), dim=-1)
            esyn = torch.norm(y_onehot - y_pred_syn, p=2, dim=1, keepdim=True)
            esem = torch.norm(y_onehot - y_pred_sem, p=2, dim=1, keepdim=True)
            esyn_norm = (esyn - esyn.min()) / (esyn.max() - esyn.min() + 1e-8)
            esem_norm = (esem - esem.min()) / (esem.max() - esem.min() + 1e-8)
            gate_input = torch.cat([esyn_norm, esem_norm, syn_out, sem_out], dim=1)
            esyn_sum = esyn.sum()
            esem_sum = esem.sum()
            grad_syn = torch.autograd.grad(esyn_sum, syn_out, retain_graph=True, create_graph=True)[0]
            grad_sem = torch.autograd.grad(esem_sum, sem_out, retain_graph=True, create_graph=True)[0]
            Hau = grad_sem * grad_syn
            final = self.opt.alpha * syn_out + self.opt.beta * sem_out + gate_input * Hau
        else:
            final = logits
        return final, kl_loss


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt)

    def forward(self, inputs):
        text_bert_indices, text_prompt_indices, aspect_bert_indices, adj_matrix, edge_adj, distance_adj, relation_adj, src_mask, aspect_mask = inputs
        h1, h2, h3, kl_loss, pooled_output = self.gcn(inputs)    # h1:16,100,768,  h2,h3:16,100,384
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)   # 16,100,384
        outputs1 = h1   # 16,768
        outputs2 = (h2 * aspect_mask).sum(dim=1) / asp_wn   # 16,384
        outputs3 = (h3 * aspect_mask).sum(dim=1) / asp_wn   # 16,384

        return outputs1, outputs2, outputs3, kl_loss, pooled_output


path = './roberta/'


class GCNBert(nn.Module):
    def __init__(self, bert, opt):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.bert_model = RobertaModel.from_pretrained(path)
        self.opt = opt
        self.layers = opt.num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = 100
        self.edge_emb = torch.load(opt.amr_edge_pt) \
            if opt.edge == "normal" or opt.edge == "same" else nn.Embedding(56000, 1024)
        self.edge_emb_layernorm = nn.LayerNorm(opt.amr_edge_dim)
        self.edge_emb_drop = nn.Dropout(opt.edge_dropout)
        self.edge_dim_change = nn.Linear(opt.amr_edge_dim, opt.hidden_dim, bias=False)

        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.wa = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.wa.append(nn.Linear(input_dim, self.mem_dim))

        self.ws = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.ws.append(nn.Linear(input_dim, self.mem_dim))

        self.linear = nn.Linear(opt.roberta_dim, opt.hidden_dim)
        self.edge_linear = nn.Linear(opt.hidden_dim, self.attdim)
        self.aspect_linear = nn.Linear(opt.bert_dim, self.attdim)
        self.lstm_linear = nn.Linear(2 * opt.hidden_dim, self.attdim)

        self.dense = nn.Linear(2 * self.attdim, opt.hidden_dim)
        self.aggregate_W = nn.Linear(2 * opt.hidden_dim, self.attdim)
        self.gc1 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.rgc1 = RelationalGraphConvLayer(5, opt.bert_dim, opt.bert_dim)
        self.rgc2 = RelationalGraphConvLayer(5, opt.bert_dim, opt.bert_dim)

        self.attn = MultiHeadAttention(self.opt, opt.attention_heads, self.bert_dim)
        self.att = PHMultiHeadAttention(opt.attention_heads, opt.hidden_dim)
        self.affine1 = nn.Parameter(torch.Tensor(opt.hidden_dim, opt.hidden_dim))
        self.affine2 = nn.Parameter(torch.Tensor(opt.hidden_dim, opt.hidden_dim))
        self.affine3 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine4 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.cnn_gat = nn.Sequential(
            nn.Conv2d(1, 10, (5, 768), (5,)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.attention = Attention(self.attdim, out_dim=self.attdim, score_function='scaled_dot_product', n_head=8, dropout=opt.dropout)
        self.text_lstm = DynamicLSTM(opt.hidden_dim, opt.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)


    def forward(self, inputs):
        text_bert_indices, text_prompt_indices, aspect_bert_indices, adj_matrix, edge_adj, distance_adj, relation_adj, src_mask, aspect_mask = inputs
        # 16,100                 16,100               16,100        16,100,100  16,100,100  16,100,100   16,5,100,100    16,100     16,100
        adj = torch.bmm(distance_adj, relation_adj.sum(dim=1))
        asp = self.bert_model(aspect_bert_indices).last_hidden_state  # 16,100,768
        aspect = self.aspect_linear(asp)  # 16,100,100
        src_mask = src_mask.unsqueeze(-2)  # 16,1,100
        batch, len, _ = edge_adj.size()  # 16 100
        text_len = torch.sum(text_bert_indices != 0, dim=-1).cpu()  # 16

        logit = self.bert(text_prompt_indices).logits    # 16,100,50265
        linear_out = self.linear(logit)  # 16,100,768

        sequence_output = self.bert_model(text_bert_indices).last_hidden_state  # 16,100,768 16,768
        sequence_output = self.layernorm(sequence_output)  # 16,100,768

        pooled_output = self.bert_model(text_bert_indices).pooler_output
        pooled_output = self.pooled_drop(pooled_output)  # 16,768
# ------------------------------------------------句法GCN----------------------------------------------------------------------------------------------
        gc1_out = F.relu(self.gc1(linear_out, adj))   # 16,100,768
        gc2_out = F.relu(self.gc2(gc1_out, adj))  # 16,100,768
        cnn_in = gc2_out.unsqueeze(1)   # 16,1,100,768
        cnn_out = self.cnn_gat(cnn_in)   # 16,10,10,1
        cnn_out = cnn_out.view(batch, -1)   # 16,100
        lstm_out, (_, _) = self.text_lstm(linear_out, text_len)
        lstm_out = self.lstm_linear(lstm_out)   # 16,83,100
        attention_output_c, _ = self.attention(cnn_out, aspect)   # 16,100,100
        attention_output_l, _ = self.attention(lstm_out, aspect)  # 16,100,100
        attention_cat = torch.cat((attention_output_c, attention_output_l), dim=-1)    # 16,100,200
        attention_mean = torch.div(torch.sum(attention_cat, dim=1), text_len.unsqueeze(1).float().cuda())  # 16,200
        outputs_ad = self.dense(attention_mean)  # 16,768

# -------------------------------------------------语义GCN--------------------------------------------------------------------------------------------
        edge_adj = self.edge_emb(edge_adj)  # 16,100,100,1024
        for layer in range(self.layers):
            edge_1 = nn.Linear(1024, self.opt.hidden_dim).to('cuda')
            edge_gcn1 = edge_1(edge_adj).sum(dim=2)  # 16,100,768
            edge_2 = nn.Linear(1024, 1).to('cuda')
            edge_gcn2 = edge_2(edge_adj).squeeze(-1)  # 16,100,100
            edge_adj = self.gc1(edge_gcn1, edge_gcn2)  # 16,100,768
        sem_input = edge_adj
        gcn_inputs = self.bert_drop(sequence_output)
        aspect_mask_resize = aspect_mask.unsqueeze(-1).repeat(1, 1, self.bert_dim)  # 16,100,768
        sem_mask = torch.ones_like(sem_input, dtype=torch.bool)
        sem_mask_con = torch.where(sem_mask, torch.ones_like(sem_input), torch.zeros_like(sem_input))
        aspect_outs = gcn_inputs * aspect_mask_resize * sem_mask_con
        aspect_scores, s_attn = self.attn(gcn_inputs, gcn_inputs, src_mask, aspect_outs, aspect_mask_resize)
        aspect_score_list = [attn_adj.squeeze(1) for attn_adj in torch.split(aspect_scores, 1, dim=1)]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(s_attn, 1, dim=1)]
        adj_ag = None
        aspect_score_avg = None
        adj_s = None
        for i in range(self.attention_heads):
            if aspect_score_avg is None:
                aspect_score_avg = aspect_score_list[i]
            else:
                aspect_score_avg += aspect_score_list[i]
        aspect_score_avg = aspect_score_avg / self.attention_heads
        for i in range(self.attention_heads):
            if adj_s is None:
                adj_s = attn_adj_list[i]
            else:
                adj_s += attn_adj_list[i]
        adj_s = adj_s / self.attention_heads
        for j in range(adj_s.size(0)):
            adj_s[j] -= torch.diag(torch.diag(adj_s[j]))
            adj_s[j] += torch.eye(adj_s[j].size(0)).cuda()  # self-loop

        ad_mask = torch.ones_like(outputs_ad[:, :1]).unsqueeze(-1)
        ad_mask = ad_mask.expand(-1, self.attdim, self.attdim)
        adj_s = src_mask.transpose(1, 2) * adj_s * ad_mask
        syn_m = torch.exp((-1.0) * self.opt.alpha * (adj_matrix + adj.mul(adj_matrix.new_zeros(1)).sum()))
        sem_m = (aspect_score_avg > torch.ones_like(aspect_score_avg) * self.opt.beta)
        syn_m = syn_m.masked_fill(sem_m, 1).cuda()
        adj_ag = (syn_m * aspect_score_avg).type(torch.float32)
        kl_loss = F.kl_div(adj_ag.softmax(-1).log(), adj_s.softmax(-1), reduction='sum')
        kl_loss = torch.exp((-1.0) * kl_loss * self.opt.gama)
        denom_s = adj_s.sum(2).unsqueeze(2) + 1
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_s = gcn_inputs
        outputs_ag = gcn_inputs
        for l in range(self.layers):
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.wa[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)
            Ax_s = adj_s.bmm(outputs_s)
            AxW_s = self.ws[l](Ax_s)
            AxW_s = AxW_s / denom_s
            gAxW_s = F.relu(AxW_s)
            A3 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine3), torch.transpose(gAxW_s, 1, 2)), dim=-1)
            A4 = F.softmax(torch.bmm(torch.matmul(gAxW_s, self.affine4), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            gAxW_ag, gAxW_s = torch.bmm(A3, gAxW_s), torch.bmm(A4, gAxW_ag)
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag   # 16,100,384
            outputs_s = self.gcn_drop(gAxW_s) if l < self.layers - 1 else gAxW_s     # 16,100,384
# ---------------------------------------------------------------------------------------------------------------------------------------------------
        return outputs_ad, outputs_ag, outputs_s, kl_loss, pooled_output
