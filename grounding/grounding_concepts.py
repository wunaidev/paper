import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import nltk
import json
import time

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
				 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
				 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
				 ])

nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]
print(nltk_stopwords)

concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
	cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]

def lemmatize(nlp, concept):

	doc = nlp(concept.replace("_"," "))
	lcs = set()
	# for i in range(len(doc)):
	#     lemmas = []
	#     for j, token in enumerate(doc):
	#         if j == i:
	#             lemmas.append(token.lemma_)
	#         else:
	#             lemmas.append(token.text)
	#     lc = "_".join(lemmas)
	#     lcs.add(lc)
	lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
	return lcs

def load_matcher(nlp):
	config = configparser.ConfigParser()
	config.read("paths.cfg")
	with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
		all_patterns = json.load(f)

	matcher = Matcher(nlp.vocab)
	for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
		matcher.add(concept, None, pattern)
	return matcher

def ground_mentioned_concepts(nlp, matcher, s, ans = ""):
	s = s.lower()
	doc = nlp(s)
	matches = matcher(doc)

	mentioned_concepts = set()
	span_to_concepts = {}

	for match_id, start, end in matches:

		span = doc[start:end].text  # the matched span
		if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
			continue
		original_concept = nlp.vocab.strings[match_id]
		# print("Matched '" + span + "' to the rule '" + string_id)

		if len(original_concept.split("_")) == 1:
			original_concept = list(lemmatize(nlp, original_concept))[0]

		if span not in span_to_concepts:
			span_to_concepts[span] = set()

		span_to_concepts[span].add(original_concept)

	for span, concepts in span_to_concepts.items():
		concepts_sorted = list(concepts)
		concepts_sorted.sort(key=len)

		# mentioned_concepts.update(concepts_sorted[0:2])

		shortest = concepts_sorted[0:3] #
		for c in shortest:
			if c in blacklist:
				continue
			lcs = lemmatize(nlp, c)
			intersect = lcs.intersection(shortest)
			if len(intersect)>0:
				mentioned_concepts.add(list(intersect)[0])
			else:
				mentioned_concepts.add(c)


	# stop = timeit.default_timer()
	# print('\t Done! Time: ', "{0:.2f} sec".format(float(stop - start_time)))

	# if __name__ == "__main__":
	#     print("Sentence: " + s)
	#     print(mentioned_concepts)
	#     print()
	return mentioned_concepts

def hard_ground(nlp, sent):
	global cpnet_vocab
	sent = sent.lower()
	doc = nlp(sent)
	res = set()
	for t in doc:
		if t.lemma_ in cpnet_vocab:
			res.add(t.lemma_)
	sent = "_".join([t.text for t in doc])
	if sent in cpnet_vocab:
		res.add(sent)
	return res

def match_mentioned_concepts(nlp, sents, answers, batch_id = -1):
	matcher = load_matcher(nlp)

	res = []
	print("Begin matching concepts.")
	st_match_time = time.time()
	for sid, s in tqdm(enumerate(sents), total=len(sents), desc="grounding batch_id:%d"%batch_id):
		a = answers[sid]
		all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
		answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
		question_concepts = all_concepts - answer_concepts
		if len(question_concepts)==0:
			# print(s)
			question_concepts = hard_ground(nlp, s) # not very possible
		if len(answer_concepts)==0:
			print(a)
			answer_concepts = hard_ground(nlp, a) # some case
			print(answer_concepts)

		res.append({"sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)})
	print("match time: {}".format(time.time()-st_match_time))
	return res

#这个是处理passage用的。现在问题是效率太低。
def match_passage_concepts(nlp, sents, batch_id = -1):
	matcher = load_matcher(nlp)

	res = []
	print("Begin matching concepts.")
	st_match_time = time.time()
	#ground_metioned_concepts太耗时。对一篇文章建模要20秒左右。根本不现实。只能用hard_ground去处理。
	#另外，过多的concepts可能也会给下一步的图构建带来压力。下一步应考虑继续更严格地裁剪concepts，使之在尽可能保留更多信息的前提下生成的图更小。
	'''
	for sid, s in tqdm(enumerate(sents), total=len(sents), desc="grounding batch_id:%d"%batch_id):
		passage_concepts = ground_mentioned_concepts(nlp, matcher, s)
		if len(passage_concepts)==0:
			print(s)
			passage_concepts = hard_ground(nlp, s)
	
		res.append({"sent": s, "ans": "", "qc": list(passage_concepts), "ac": []})
	'''
	for sid, s in tqdm(enumerate(sents), total=len(sents), desc="grounding batch_id:%d"%batch_id):
		passage_concepts = hard_ground(nlp, s)
		res.append({"sent": s, "ans": "", "qc": list(passage_concepts), "ac": []})
	print("match time: {}".format(time.time()-st_match_time))
	return res


def process(filename, batch_id=-1):


	nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
	nlp.add_pipe(nlp.create_pipe('sentencizer'))

	sents = []
	answers = []
	with open(filename, 'r') as f:
		lines = f.read().split("\n")


	for line in tqdm(lines, desc="loading file"):
		if line == "":
			continue
		j = json.loads(line)
		for statement in j["statements"]:
			sents.append(statement["statement"])
		for answer in j["question"]["choices"]:
			answers.append(answer["text"])


	if batch_id >= 0:
		output_path = filename + ".%d.mcp" % batch_id
		batch_sents = list(np.array_split(sents, 100)[batch_id])
		batch_answers = list(np.array_split(answers, 100)[batch_id])
	else:
		output_path = filename + ".mcp"
		batch_sents = sents
		batch_answers = answers

	res = match_mentioned_concepts(nlp, sents=batch_sents, answers=batch_answers, batch_id=batch_id)
	with open(output_path, 'w') as fo:
		json.dump(res, fo)



def test():
	def test_ground(nlp):
		
		passage = '''One thinks of princes and presidents as some of the most powerful people in the world; however, governments, elected or otherwise, sometimes have had to struggle with the financial powerhouses called tycoons. The word tycoon is relatively new to the English language. It is Chinese in origin but was given as a title to some Japanese generals. The term was brought to the United States, in the late nineteenth century, where it eventually was used to refer to magnates who acquired immense fortunes from sugar and cattle, coal and oil, rubber and steel, and railroads. Some people called these tycoons "capitals of industry" and praised them for their contributions to U.S. wealth and international reputation. Others criticized them as cruel "robber barons", who would stop at nothing in pursuit of personal wealth.
					The early tycoons built successful businesses, often taking over smaller companies to eliminate competition. A single company that came to control an entire market was called a monopoly. Monopolies made a few families very wealthy, but they also placed a heavy financial burden on consumers and the economy at large.
					As the country expanded and railroads linked the East Coast to the West Coast, local monopolies turned into national corporations called trusts. A trust is a group of companies that join together under the control of a board of trustees. Railroad trusts are an excellent example. Railroads were privately owned and operated and often monopolized various routes, setting rates as high as they desired. The financial burden this placed on passengers and businesses increased when railroads formed trusts. Farmers, for example, had no choice but to pay, as railroads were the only means they could use to get their grain to buyers. Exorbitant   goods rates put some farmers out of business.
					There were even accusations that the trusts controlled government itself by buying votes and manipulating elected officials. In 1890 Congress passed the Sherman Antitrust. Act, legislation aimed at breaking the power of such trusts. The Sherman Antitrust Act focused on two main issues. First of all, it made illegal any effort to interfere with the normal conduct of interstate trade. It also made it illegal to monopolize any part of business that operates across state lines.
					Over the next 60 years or so, Congress passed other antitrust laws in an effort to encourage competition and restrict the power of larger corporations.'''
		#res = match_mentioned_concepts(nlp, sents=["The Sherman Antitrust Act affected only the companies doing business within state lines."], answers=["affected only the companies doing business within state lines"])
		#res = match_mentioned_concepts(nlp, sents=["One thinks of princes and presidents as some of the most powerful people in the world; however, governments, elected or otherwise, sometimes have had to struggle with the financial powerhouses called tycoons."], answers=[""])
		#res = match_mentioned_concepts(nlp, sents=[passage], answers=[""])

		res = match_passage_concepts(nlp, sents=[passage])
		#res = match_passage_concepts(nlp, sents=passage.split("."))
		print(res)
		print(len(res[0]['qc']))
		return res
	
	def test_trune(data):
		concept_vocab = set()

		config = configparser.ConfigParser()
		config.read("paths.cfg")
		with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
			cpnet_vocab = set([l.strip() for l in list(f.readlines())])

		prune_data = []
		for item in tqdm(data):
			qc = item["qc"]
			prune_qc = []
			for c in qc:
				if c[-2:] == "er" and c[:-2] in qc:
					continue
				if c[-1:] == "e" and c[:-1] in qc:
					continue
				have_stop = False
				for t in c.split("_"):
					if t in nltk_stopwords:
						have_stop = True
				if not have_stop and c in cpnet_vocab:
					prune_qc.append(c)

			ac = item["ac"]
			prune_ac = []
			for c in ac:
				if c[-2:] == "er" and c[:-2] in ac:
					continue
				if c[-1:] == "e" and c[:-1] in ac:
					continue
				all_stop = True
				for t in c.split("_"):
					if t not in nltk_stopwords:
						all_stop = False
				if not all_stop and c in cpnet_vocab:
					prune_ac.append(c)

			item["qc"] = prune_qc
			item["ac"] = prune_ac

			prune_data.append(item)
			print("***")
			print(prune_data)
			print(print(len(prune_data[0]['qc'])))

	print("loading nlp...")
	nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
	nlp.add_pipe(nlp.create_pipe('sentencizer'))

	print("start concept ground...")
	data = test_ground(nlp)
	pre_time = time.time()
	print("start trune...")
	test_trune(data)
	print("trune time:{}".format(time.time()-pre_time))



# "sent": "Watch television do children require to grow up healthy.", "ans": "watch television",
if __name__ == "__main__":
	process(sys.argv[1], int(sys.argv[2]))
	
	#test()
