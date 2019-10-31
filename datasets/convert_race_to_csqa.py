import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
#import apex
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

#from transformers import BertTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class RaceExample(object):
	"""A single training/test example for the RACE dataset."""
	'''
	For RACE dataset:
	race_id: data id
	context_sentence: article
	start_ending: question
	ending_0/1/2/3: option_0/1/2/3
	label: true answer
	'''
	def __init__(self,
				 race_id,
				 context_sentence,
				 start_ending,
				 ending_0,
				 ending_1,
				 ending_2,
				 ending_3,
				 label = None):
		self.race_id = race_id
		self.context_sentence = context_sentence
		self.start_ending = start_ending
		self.endings = [
			ending_0,
			ending_1,
			ending_2,
			ending_3,
		]
		self.label = label

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		l = [
			f"id: {self.race_id}",
			f"article: {self.context_sentence}",
			f"question: {self.start_ending}",
			f"option_0: {self.endings[0]}",
			f"option_1: {self.endings[1]}",
			f"option_2: {self.endings[2]}",
			f"option_3: {self.endings[3]}",
		]

		if self.label is not None:
			l.append(f"label: {self.label}")

		return ", ".join(l)



class InputFeatures(object):
	def __init__(self,
				 example_id,
				 choices_features,
				 label

	):
		self.example_id = example_id
		self.choices_features = [
			{
				'input_ids': input_ids,
				'input_mask': input_mask,
				'segment_ids': segment_ids
			}
			for _, input_ids, input_mask, segment_ids in choices_features
		]
		self.label = label




## paths is a list containing all paths
def read_race_examples(paths):
	examples = []
	for path in paths:
		filenames = glob.glob(path+"/*txt")
		for filename in filenames:
			with open(filename, 'r', encoding='utf-8') as fpr:
				data_raw = json.load(fpr)
				article = data_raw['article']
				## for each qn
				for i in range(len(data_raw['answers'])):
					truth = ord(data_raw['answers'][i]) - ord('A')
					question = data_raw['questions'][i]
					options = data_raw['options'][i]
					examples.append(
						RaceExample(
							race_id = filename+'-'+str(i),
							context_sentence = article,
							start_ending = question,

							ending_0 = options[0],
							ending_1 = options[1],
							ending_2 = options[2],
							ending_3 = options[3],
							label = truth))
				
	return examples 



def convert_examples_to_features(examples, tokenizer, max_seq_length,
								 is_training):
	"""Loads a data file into a list of `InputBatch`s."""

	# RACE is a multiple choice task. To perform this task using Bert,
	# we will use the formatting proposed in "Improving Language
	# Understanding by Generative Pre-Training" and suggested by
	# @jacobdevlin-google in this issue
	# https://github.com/google-research/bert/issues/38.
	#
	# The input will be like:
	# [CLS] Article [SEP] Question + Option [SEP]
	# for each option 
	# 
	# The model will output a single value for each input. To get the
	# final decision of the model, we will run a softmax over these 4
	# outputs.
	features = []
	for example_index, example in enumerate(examples):
		context_tokens = tokenizer.tokenize(example.context_sentence)
		start_ending_tokens = tokenizer.tokenize(example.start_ending)

		choices_features = []
		for ending_index, ending in enumerate(example.endings):
			# We create a copy of the context tokens in order to be
			# able to shrink it according to ending_tokens
			context_tokens_choice = context_tokens[:]
			ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
			# Modifies `context_tokens_choice` and `ending_tokens` in
			# place so that the total length is less than the
			# specified length.  Account for [CLS], [SEP], [SEP] with
			# "- 3"
			_truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

			tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
			segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			padding = [0] * (max_seq_length - len(input_ids))
			input_ids += padding
			input_mask += padding
			segment_ids += padding

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length

			choices_features.append((tokens, input_ids, input_mask, segment_ids))

		label = example.label
		## display some example
		if example_index < 1:
			logger.info("*** Example ***")
			logger.info(f"race_id: {example.race_id}")
			for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
				logger.info(f"choice: {choice_idx}")
				logger.info(f"tokens: {' '.join(tokens)}")
				logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
				logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
				logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
			if is_training:
				logger.info(f"label: {label}")

		features.append(
			InputFeatures(
				example_id = example.race_id,
				choices_features = choices_features,
				label = label
			)
		)

	return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs == labels)

def select_field_single(feature, field):
	return[
			choice[field]
			for choice in feature.choices_features
		]

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x






'''
把race格式转换成csqa格式
convert race format to csqa format

Input race：list(RaceExample)
race_id: data id
context_sentence: article
start_ending: question
ending_0/1/2/3: option_0/1/2/3
label: true answer

Output csqa(JSONL)：
JSONL format of files
 {
   "id": "d3b479933e716fb388dfb297e881054c",
   "question": {
	  "stem": "If a lantern is not for sale, where is it likely to be?"
	  "choices": [{"label": "A", "text": "antique shop"}, {"label": "B", "text": "house"}, {"label": "C", "text": "dark place"}]
	},
	"answerKey":"B"
	"passage"? : ""?
	"InputFeatures":{"input_ids"? : []? ,"input_mask"? : []?, "segment_ids"? : []?}
}
'''
def race2csqa(race_list : list, features_list : list) -> dict:
	def example2csqa(example : RaceExample, features: InputFeatures):
		csqa_dict = dict()
		csqa_dict["id"] = example.race_id
		#csqa dont have passages
		csqa_dict["passage"] = example.context_sentence
		csqa_dict["question"] = {"stem": example.start_ending, "choices": [{"label": chr(ord(str(index)) + ord('A') - ord('0')), "text": ending} for index, ending in enumerate(example.endings)]}
		csqa_dict["answerKey"] = chr(ord(str(example.label)) + ord('A') - ord('0'))
		
		'''usage:
			features = csqa_dict["InputFeatures"]
			input_mask = features["input_mask"]
			input_ids = features["input_ids"]
			segment_ids = features["segment_ids"]
			labels = features["labels"]
		'''
		
		csqa_dict["InputFeatures"] = {"input_ids": select_field_single(features, 'input_ids'),
									"input_mask": select_field_single(features, 'input_mask'),
									"segment_ids": select_field_single(features, 'segment_ids'),
									"labels": features.label
									}
		
		return csqa_dict

	csqa_list = list(map(example2csqa, race_list, features_list))
	print(csqa_list[:5])

	return csqa_list


'''
把*格式转换*(race2csqa)后的race数据保存为jsonl
'''
def save_as_jsonl(race_list : list, outfile_path : str) -> None:
	with open(outfile_path, 'w') as of:
		for line in race_list:
			of.write(json.dumps(line))
			of.write('\n')
	print("Done! save as {}".format(outfile_path))




def process(work_method : str, DATA_DIR : str):
	MAX_SEQ_LENGTH = 380
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	
	outfile_path = r"../datasets/csqa_new/{}_rand_split.jsonl".format(work_method)
	data_dir = os.path.join(DATA_DIR, work_method)
	data_examples = read_race_examples([data_dir+'/high', data_dir+'/middle'])
	#example = train_examples[:5]
	data_features = convert_examples_to_features(
			data_examples, tokenizer, MAX_SEQ_LENGTH, True)
	
	save_as_jsonl(race2csqa(data_examples, data_features), outfile_path)



if __name__=='__main__':
	if len(sys.argv) < 2:
		raise ValueError("Provide at least one arguments: race data file")
	DATA_DIR = sys.argv[1]
	process(work_method='dev', DATA_DIR=DATA_DIR)
	process(work_method='test', DATA_DIR=DATA_DIR)
	process(work_method='train', DATA_DIR=DATA_DIR)
