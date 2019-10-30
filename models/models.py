import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from torch.nn import init
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from dgl import DGLGraph
import random

from transformers import BertForMultipleChoice
from transformers import BertPreTrainedModel
from transformers import BertModel
from torch import nn

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


BERT_EMBEDDING_DIM = 300



def weight_init(m):
	'''
	Usage:
		model = Model()
		model.apply(weight_init)
	'''
	if isinstance(m, nn.Conv1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm1d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm3d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.LSTM):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.LSTMCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRU):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRUCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)



class RelationNetwork(nn.Module):
	def __init__(self, concept_dim, concept_num, pretrained_concept_emd, sent_dim, latent_rel_dim, device):

		super(RelationNetwork, self).__init__()

		self.concept_dim = concept_dim
		# self.context_dim = context_dim # qas triple vector
		self.sent_dim = sent_dim
		self.concept_emd = nn.Embedding(concept_dim, concept_num)

		# random init the embeddings 
		if pretrained_concept_emd is not None:
			# self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
			self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)
		else:
			bias = np.sqrt(6.0 / self.concept_dim)
			nn.init.uniform_(self.concept_emd.weight, -bias, bias)
		self.latent_rel_dim = latent_rel_dim
		self.device = device
		self.relation_extractor = nn.Sequential(
			nn.Linear(2*concept_dim+sent_dim, self.latent_rel_dim*2),  # binary classification
			nn.ReLU(),
			nn.BatchNorm1d(self.latent_rel_dim*2),
			nn.Linear(self.latent_rel_dim*2, self.latent_rel_dim),
			nn.BatchNorm1d(self.latent_rel_dim),
			nn.ReLU(),
		)
		self.hidden2output = nn.Sequential(
			nn.Linear(latent_rel_dim, 1),  # binary classification
			nn.Sigmoid()
		)


	# qas_vec is the concat of the question concept, answer concept, and the statement
	def forward(self, statement_vecs, qa_pairs):

		pooled_qas_vecs_list = []
		for index in range(len(statement_vecs)):  # len = batch_size * num_choices
			s_vec = statement_vecs[index].to(self.device)
			if len(qa_pairs[index]) == 0 or False: # if True then abaliate qa pairs
				qas_vecs = torch.cat((torch.zeros(1, self.concept_dim).to(self.device),
									   torch.zeros(1, self.concept_dim).to(self.device),
									   torch.stack([s_vec]).to(self.device)), dim=1).to(self.device)
			else:
				q_seq = []
				a_seq = []
				for qa_pair in qa_pairs[index]:
					q, a = qa_pair[0], qa_pair[1]
					q_seq.append(q)
					a_seq.append(a)

				q_seq = torch.LongTensor(q_seq).to(self.device)
				a_seq = torch.LongTensor(a_seq).to(self.device)
				q_vecs = self.concept_emd(q_seq).to(self.device)
				a_vecs = self.concept_emd(a_seq).to(self.device)
				s_vecs = torch.stack([s_vec] * len(qa_pairs[index])).to(self.device)
				qas_vecs = torch.cat((q_vecs, a_vecs, s_vecs), dim=1)


			pooled_qas_vecs = qas_vecs.mean(dim=0).to(self.device)  # mean/sum/attention
			
			pooled_qas_vecs_list.append(pooled_qas_vecs)
		latent_rel_vecs = self.relation_extractor(torch.stack(pooled_qas_vecs_list))
		logits = self.hidden2output(latent_rel_vecs).to(self.device)
		# logits = torch.stack(logits).to(self.device)
		return logits


class KnowledgeEnhancedRelationNetwork(nn.Module):
	def __init__(self, sent_dim, concept_dim, relation_dim,
				 concept_num, relation_num, qas_encoded_dim,
				 pretrained_concept_emd, pretrained_relation_emd,
				 lstm_dim, lstm_layer_num, device,
				 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True, qa_attention=True
				 ): 

		super(KnowledgeEnhancedRelationNetwork, self).__init__()
		self.num_random_paths = num_random_paths
		self.concept_dim = concept_dim
		self.relation_dim = relation_dim
		self.path_attention = path_attention
		self.qa_attention = qa_attention

		self.sent_dim = sent_dim
		self.concept_emd = nn.Embedding(concept_dim, concept_num)
		self.relation_emd = nn.Embedding(relation_num, relation_dim)



		# random init the embeddings
		if pretrained_concept_emd is not None:
			# self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
			self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)
		else:
			bias = np.sqrt(6.0 / self.concept_dim)
			nn.init.uniform_(self.concept_emd.weight, -bias, bias)

		if pretrained_relation_emd is not None:
			# self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
			self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
		else:
			bias = np.sqrt(6.0 / self.relation_dim)
			nn.init.uniform_(self.relation_emd.weight, -bias, bias)

		self.qas_encoded_dim = qas_encoded_dim



		self.lstm = nn.LSTM(input_size=concept_dim+relation_dim,
							hidden_size=lstm_dim,
							num_layers=lstm_layer_num,
							bidirectional=bidirect,
							dropout=dropout
							)

		if bidirect:
			self.lstm_dim = lstm_dim * 2
		else:
			self.lstm_dim = lstm_dim

		self.qas_encoder = nn.Sequential(
			nn.Linear(2 * concept_dim+sent_dim, self.qas_encoded_dim * 2),  # binary classification
			nn.Dropout(dropout),
			nn.LeakyReLU(),
			nn.Linear(self.qas_encoded_dim * 2, self.qas_encoded_dim),
			nn.Dropout(dropout),
			nn.LeakyReLU(),
		)


		self.nonlinear = nn.LeakyReLU()
		if self.path_attention:
			self.qas_pathlstm_att = nn.Linear(self.qas_encoded_dim, self.lstm_dim)  # transform qas vector to query vectors
			self.qas_pathlstm_att.apply(weight_init)

		if self.qa_attention:
			self.sent_ltrel_att = nn.Linear(sent_dim, self.qas_encoded_dim) # transform sentence vector to query vectors
			self.sent_ltrel_att.apply(weight_init)

		self.device = device
		self.hidden2output = nn.Sequential(
			nn.Linear(self.qas_encoded_dim + self.lstm_dim + self.sent_dim + 300, 1),  # binary classification
			nn.Sigmoid()
		)

		self.lstm.apply(weight_init)
		self.qas_encoder.apply(weight_init)
		self.hidden2output.apply(weight_init)


	def paths_group(self, cpt_paths, rel_paths, q, a, k=None):
		qa_cpt_paths = []
		qa_rel_paths = []
		assert len(cpt_paths) == len(rel_paths)
		for index, p in enumerate(cpt_paths):
			end = 0
			for t in p[::-1]:
				if t != 0:
					end = t
					break
			if p[0] == q and end == a:
				qa_cpt_paths.append(p)
				qa_rel_paths.append(rel_paths[index])

		if not self.training or k is None or k < 0:  # in eval mode, we can use all the paths, which is more time-consuming
			return qa_cpt_paths, qa_rel_paths

		# assert len(qa_cpt_paths) > 0
		random_index = list(range(len(qa_cpt_paths)))
		random.shuffle(random_index)
		random_qa_cpt_paths = []
		random_qa_rel_paths = []
		for index in random_index[:k]:
			random_qa_cpt_paths.append(qa_cpt_paths[index])
			random_qa_rel_paths.append(qa_rel_paths[index])
		return random_qa_cpt_paths, random_qa_rel_paths


	# qas_vec is the concat of the question concept, answer concept, and the statement
	def forward(self, s_vec_batched, qa_pairs_batched, cpt_paths_batched, rel_paths_batched, ana_mode=False):
		self.device = self.concept_emd.weight.device  # multiple GPUs need to specify device
		final_vecs = []




		if ana_mode:
			path_att_scores = []
			qa_pair_att_scores = []



		for index in range(len(s_vec_batched)):  # len = batch_size * num_choices
			# for each question-answer statement

			s_vec = s_vec_batched[index].to(self.device)
			cpt_paths = cpt_paths_batched[index]
			rel_paths = rel_paths_batched[index]


			if len(qa_pairs_batched[index]) == 0 or False: # if "or True" then we can do abalation study
				raw_qas_vecs = torch.cat((torch.zeros(1, self.concept_dim).to(self.device),
									  torch.zeros(1, self.concept_dim).to(self.device),
									  torch.stack([s_vec]).to(self.device)
									  ), dim=1).to(self.device)

				qas_vecs = self.qas_encoder(raw_qas_vecs)
				# print("0:", qas_vecs.size())
				latent_rel_vecs = torch.cat((qas_vecs, torch.zeros(1, self.lstm_dim).to(self.device)), dim=1)
			else:
				q_seq = []
				a_seq = []

				qa_path_num = []

				tmp_cpt_paths = []
				for qa_pair in qa_pairs_batched[index]:  # for each possible qc, ac pair
					q, a = qa_pair[0], qa_pair[1]
					q_seq.append(q)
					a_seq.append(a)

					qa_cpt_paths, qa_rel_paths = self.paths_group(cpt_paths, rel_paths, q, a, k=self.num_random_paths) # self.num_random_paths

					qa_path_num.append(len(qa_cpt_paths))
					tmp_cpt_paths.extend(qa_cpt_paths)

					# assert that the order is contiunous
				if self.num_random_paths is None:
					assert tmp_cpt_paths == cpt_paths

				q_seq = torch.LongTensor(q_seq).to(self.device)
				a_seq = torch.LongTensor(a_seq).to(self.device)
				q_vecs = self.concept_emd(q_seq)
				a_vecs = self.concept_emd(a_seq)

				# q_vecs = q_vecss[index] # self.concept_emd(q_seq)
				# a_vecs = a_vecss[index] # self.concept_emd(a_seq)

				s_vecs = torch.stack([s_vec] * len(qa_pairs_batched[index]))
				raw_qas_vecs = torch.cat((q_vecs, a_vecs, s_vecs), dim=1)
				# all the qas triple vectors associated with a statement

				qas_vecs = self.qas_encoder(raw_qas_vecs)

				# print(qas_vecs.size())
				# print(len(all_qa_cpt_paths_embeds))

				pooled_path_vecs = []


				# batched path encoding
				batched_all_qa_cpt_paths_embeds = self.concept_emd(torch.LongTensor(cpt_paths).to(self.device)).permute(1, 0, 2)
				batched_all_qa_rel_paths_embeds = self.relation_emd(torch.LongTensor(rel_paths).to(self.device)).permute(1, 0, 2)

				batched_all_qa_cpt_rel_path_embeds = torch.cat((batched_all_qa_cpt_paths_embeds, batched_all_qa_rel_paths_embeds), dim=2)
				#

				# batched_all_qa_cpt_rel_path_embeds = over_batched_all_qa_cpt_rel_path_embeds[0:None,path_splits[index][0]:path_splits[index][1],0:None]
				# if False then abiliate the LSTM
				if True:
					batched_lstm_outs, _ = self.lstm(batched_all_qa_cpt_rel_path_embeds)
				else:
					batched_lstm_outs = torch.zeros(batched_all_qa_cpt_rel_path_embeds.size()[0],
													batched_all_qa_cpt_rel_path_embeds.size()[1],
													self.lstm_dim
													).to(self.device)

				if self.path_attention:
					query_vecs = self.qas_pathlstm_att(qas_vecs)

				cur_start = 0
				for index in range(len(qa_path_num)):
					if self.path_attention:
						query_vec = query_vecs[index]
					cur_end = cur_start + qa_path_num[index]

					# mean_pooled_path_vec = batched_lstm_outs[-1, cur_start:cur_end, :].mean(dim=0)  # mean pooling
					# attention pooling
					blo = batched_lstm_outs[-1, cur_start:cur_end, :]
					if self.path_attention:
						att_scores = torch.mv(blo, query_vec) # path-level attention scores
						norm_att_scores = F.softmax(att_scores, dim=0)
						att_pooled_path_vec = torch.mv(torch.t(blo), norm_att_scores)
						if ana_mode:
							path_att_scores.append(norm_att_scores)
					else:
						att_pooled_path_vec = blo.mean(dim=0)

					cur_start = cur_end
					pooled_path_vecs.append(att_pooled_path_vec)

				pooled_path_vecs = torch.stack(pooled_path_vecs)
				latent_rel_vecs = torch.cat((qas_vecs, pooled_path_vecs), dim=1)  # qas and KE-qas

			# final_vec = latent_rel_vecs.mean(dim=0).to(self.device)  # mean pooling

			# att pooling
			if self.path_attention:
				sent_as_query = self.sent_ltrel_att(s_vec) # sent attend on qas
				r_att_scores = torch.mv(qas_vecs, sent_as_query)  # qa-pair-level attention scores
				norm_r_att_scores = F.softmax(r_att_scores, dim=0)
				if ana_mode:
					qa_pair_att_scores.append(norm_r_att_scores)

				final_vec = torch.mv(torch.t(latent_rel_vecs), norm_r_att_scores)

			else:
				final_vec = latent_rel_vecs.mean(dim=0).to(self.device)  # mean pooling

			final_vecs.append(torch.cat((final_vec, s_vec), dim=0))

		logits = self.hidden2output(torch.stack(final_vecs))
		if not ana_mode:
			return logits
		else:
			return logits, path_att_scores, qa_pair_att_scores


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		h = self.activation(h)
		return {'h' : h}


class GraphConvLayer(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(GraphConvLayer, self).__init__()
		self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(gcn_msg, gcn_reduce)
		g.apply_nodes(func=self.apply_mod)
		return g.ndata.pop('h')




class GCN_Encoder(nn.Module):
	def __init__(self, concept_dim, hidden_dim, output_dim, pretrained_concept_emd, concept_emd=None):
		super(GCN_Encoder, self).__init__()

		self.gcn1 = GraphConvLayer(concept_dim, hidden_dim, F.relu)
		self.gcn2 = GraphConvLayer(hidden_dim, output_dim, F.relu)

		if pretrained_concept_emd is not None and concept_emd is None:
			self.concept_emd = nn.Embedding(num_embeddings=pretrained_concept_emd.size(0),
										embedding_dim=pretrained_concept_emd.size(1))
			self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)  # init
		elif pretrained_concept_emd is None and concept_emd is not None:
			self.concept_emd = concept_emd

	def forward(self, g):
		features = self.concept_emd(g.ndata["cncpt_ids"])
		x = self.gcn1(g, features)
		x = self.gcn2(g, x)
		g.ndata['h'] = x
		return g

class GCN_Sent(nn.Module):

	def __init__(self, sent_dim, sent_hidden_dim, concept_dim, graph_hidden_dim, graph_output_dim, pretrained_concept_emd, dropout=0.3):
		super(GCN_Sent, self).__init__()

		self.graph_encoder = \
			GCN_Encoder(concept_dim, graph_hidden_dim, graph_output_dim, pretrained_concept_emd)
		self.sent_dim = sent_dim
		self.sent_hidden = sent_hidden_dim
		self.MLP = nn.Sequential(
			nn.Linear(self.sent_dim+graph_output_dim, self.sent_hidden * 4),
			# nn.Linear(self.sent_dim, self.sent_hidden * 4),
			nn.BatchNorm1d(self.sent_hidden * 4),
			nn.ReLU(),
			nn.Dropout(dropout),

			nn.Linear(self.sent_hidden * 4, self.sent_hidden),
			nn.BatchNorm1d(self.sent_hidden),
			nn.ReLU(),
			nn.Dropout(dropout),

			nn.Linear(self.sent_hidden, 1),
			nn.Sigmoid()
		)

	def forward(self, sent_vecs, g):
		out_graph = self.graph_encoder(g)
		graph_vecs = dgl.mean_nodes(out_graph, 'h')
		cat = torch.cat((sent_vecs, graph_vecs), dim=1)
		# cat = sent_vecs
		logits = self.MLP(cat)
		return logits



class KnowledgeAwareGraphNetworks(nn.Module):
	def __init__(self, sent_dim, concept_dim, relation_dim,
				 concept_num, relation_num, qas_encoded_dim,
				 pretrained_concept_emd, pretrained_relation_emd,
				 lstm_dim, lstm_layer_num, device, graph_hidden_dim, graph_output_dim,
				 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True, qa_attention=True
				 ):

		super(KnowledgeAwareGraphNetworks, self).__init__()
		self.num_random_paths = num_random_paths
		self.concept_dim = concept_dim
		self.relation_dim = relation_dim
		self.path_attention = path_attention
		self.qa_attention = qa_attention

		self.sent_dim = sent_dim
		self.concept_emd = nn.Embedding(concept_dim, concept_num)
		self.relation_emd = nn.Embedding(relation_num, relation_dim)
		self.graph_hidden_dim = graph_hidden_dim
		self.graph_output_dim = graph_output_dim


		# random init the embeddings
		if pretrained_concept_emd is not None:
			# self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
			self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)
		else:
			bias = np.sqrt(6.0 / self.concept_dim)
			nn.init.uniform_(self.concept_emd.weight, -bias, bias)

		if pretrained_relation_emd is not None:
			# self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
			self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
		else:
			bias = np.sqrt(6.0 / self.relation_dim)
			nn.init.uniform_(self.relation_emd.weight, -bias, bias)

		self.qas_encoded_dim = qas_encoded_dim



		self.lstm = nn.LSTM(input_size=self.graph_output_dim + self.concept_dim + self.relation_dim,
							hidden_size=lstm_dim,
							num_layers=lstm_layer_num,
							bidirectional=bidirect,
							dropout=dropout
							)

		if bidirect:
			self.lstm_dim = lstm_dim * 2
		else:
			self.lstm_dim = lstm_dim

		self.qas_encoder = nn.Sequential(
			nn.Linear(2 * (self.graph_output_dim + self.concept_dim) + sent_dim, self.qas_encoded_dim * 2),  # binary classification
			nn.Dropout(dropout),
			nn.LeakyReLU(),
			nn.Linear(self.qas_encoded_dim * 2, self.qas_encoded_dim),
			nn.Dropout(dropout),
			nn.LeakyReLU(),
		)


		self.nonlinear = nn.LeakyReLU()
		if self.path_attention:
			self.qas_pathlstm_att = nn.Linear(self.qas_encoded_dim, self.lstm_dim)  # transform qas vector to query vectors
			self.qas_pathlstm_att.apply(weight_init)

		if self.qa_attention:
			self.sent_ltrel_att = nn.Linear(sent_dim, self.qas_encoded_dim) # transform sentence vector to query vectors
			self.sent_ltrel_att.apply(weight_init)

		self.device = device
		self.hidden2output = nn.Sequential(
			nn.Linear(self.qas_encoded_dim + self.lstm_dim + self.sent_dim + 300, 1),  # binary classification
			nn.Sigmoid()
		)

		self.lstm.apply(weight_init)
		self.qas_encoder.apply(weight_init)
		self.hidden2output.apply(weight_init)

		self.graph_encoder = GCN_Encoder(self.concept_dim, self.graph_hidden_dim, self.graph_output_dim,
										 pretrained_concept_emd=None, concept_emd=self.concept_emd)

		self.bert_multiple_choice = BertForMultipleChoice_with_logits.from_pretrained('bert-base-uncased')
		
		self.qa_linear = torch.nn.Linear(300, 768)
		self.pas_linear = torch.nn.Linear(300, 768)
		self.attn_qa = torch.nn.MultiheadAttention(768, 8)
		self.attn_pas = torch.nn.MultiheadAttention(768, 8)

		self.classifier = torch.nn.Linear(768, 1)



	# qas_vec is the concat of the question concept, answer concept, and the statement
	# s_vec_batched:list, len==len(statements) * num_choice

	def forward(self, s_vec_batched, graphs, input_ids, segment_ids, input_mask, labels=None, ana_mode=False):
		self.device = self.concept_emd.weight.device  # multiple GPUs need to specify device
		final_vecs = []
		output_graphs = self.graph_encoder(graphs)
		output_concept_embeds = torch.cat((output_graphs.ndata["h"], torch.zeros(1, self.graph_output_dim).to(self.device))) # len(output_concept_embeds) as padding


		# new_concept_embed = nn.Embedding(output_concept_embeds.size()[0], output_concept_embeds.size()[1])
		# new_concept_embed.weight = nn.Parameter(output_concept_embeds)

		new_concept_embed = torch.cat((output_graphs.ndata["h"], s_vec_batched.new_zeros((1, self.graph_output_dim))))
		new_concept_embed = new_concept_embed.to(self.device)

		#ϣ���ǣ�[BATCHSIZE, nodes, EMBEDDING]
		print("graph_size: {}".format(new_concept_embed.size()))

		logits = self.bert_multiple_choice(input_ids, segment_ids, input_mask)[0].to(self.device)
		#[BATCHSIZE, 4, EMBEDDING]
		print("bert_size: {}".format(logits.size()))

		new_concept_embed = self.qa_linear(new_concept_embed)

		#[BATCHSIZE, 4, EMBEDDING]
		logits = self.attn_qa(logits, new_concept_embed, new_concept_embed)


		#[BATCHSIZE, 4, 1]
		logits = self.classifier(logits)
		
		#[BATCHSIZE, 4]
		logits = logits.view(-1, 4)

		outputs = (logits,)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(reshaped_logits, labels)
			outputs = (loss,) + outputs

		return logits




class BertForMultipleChoice_with_logits(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMultipleChoice_with_logits, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]


        pooled_output = self.dropout(pooled_output)
        reshaped_logits = pooled_output.view(-1, num_choices, pooled_output.size(-1))
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)