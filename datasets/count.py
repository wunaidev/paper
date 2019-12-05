import re
import os

file_list = []
def search_file(search_dir='./'):
    global file_list
    all_list = os.listdir(search_dir)
    for file in all_list:
        file_path = os.path.join(search_dir, file)
        if os.path.isfile(file_path):
            if file.endswith(".txt"):
                file_list.append(file_path)
        if os.path.isdir(file_path):
            search_file(file_path)

search_file()
length_list = []
for file in file_list:
    with open(file, 'r') as f:
            txt = f.readlines()[0]
            passage = txt.split(r'"article": "')[-1].split(r'", "id": ')[0]
            words = passage.split(" ")
            length = len(words)
            length_list.append(length)

sum = 0
max_len = -1
for length in length_list:
    sum += length
    max_len = max_len if max_len > length else length


avg = sum/len(length_list)

print("we have {} reading passages.".format(len(length_list)))
from itertools import groupby
for k, g in groupby(sorted(length_list), key=lambda x: x//50):
    print('{}-{}: {}'.format(k*50, (k+1)*50-1, len(list(g))))
print("avg: {}, max: {}".format(avg, max_len))
