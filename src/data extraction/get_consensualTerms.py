# This simple Python script extracts consensual term spans from the unified version of the ACL RD-TEC 2.0.
# The script reads a unified XML file and outputs consensual term spans with semantic class information to 2 categories of files.
# The script needs to iterate over files. I did this through batch, so the loop is not present here. Add it if need be.

from __future__ import print_function #use python3 printing function in python 2
import sys
import xml.etree.ElementTree as ET
import re

infile = open(sys.argv[1], 'r')
outfile = open('OUTPUTPATH/allConsensualTerms_classes.txt', 'a')
outfile2 = open('OUTPUTPATH/allConsensualSpans_differentClasses.txt', 'a')

tree = ET.parse(infile)
root = tree.getroot()

for sent in root.iter('S'):
	terms = sent.findall('term')
	for term in terms:
		tokens = term.findall('token')
		tokentext = []
		for token in tokens:
			tokentext.append(token.text)
		output = ' '.join(tokentext)
		class1 = term.get('class.1')
		class2 = term.get('class.2')
		if class1==class2:
			match = re.search('^(None)', str(class1)) #ignore common Nones
			if not match:
				print(output, class1, class2, sep='\t', file=outfile, end="\n")
		else:
			match1 = re.search('^(None)', str(class1)) #ignore cases with None class
			match2 = re.search('^(None)', str(class2))
			if not match1:
				if not match2:
					print(output, class1, class2, sep='\t', file=outfile2, end='\n')

infile.close()
outfile.close()
outfile2.close()