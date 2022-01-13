import json

#Clear existing files before sorting json files
for x in range(1935,2020):
	filename = str(x) + '.txt'
	open(filename, 'w').close()

#For use with the V10 DBLP database from https://www.aminer.org/citation
#This data set is split into four .json files, which are imported in sequence
#Note: this takes a long time to run, so it's possible to comment out multiple sections of the data set for quicker running and testing
file = open('dblp-ref-0.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

file = open('dblp-ref-1.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

file = open('dblp-ref-2.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

file = open('dblp-ref-3.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()