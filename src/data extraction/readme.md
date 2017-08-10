# Readme

This file describes the contents of this folder. The folder holds a set of ad-hoc Perl and Python scripts that were used to extract the term data for training and classification from ACL RD-TEC. Scripts are put here for documentation rather than active development.

1. get_consensualTerms.py
This script processes [unified XML from ACL RD-TEC 2.0](https://github.com/languagerecipes/acl-rd-tec-2.0/tree/master/distribution/annoitation_files/double_annotated_files/unified_xml) to extract two types of semantic class annotations:
	* term spans with identical class annotations
	* term spans with diverging class annotations
Make sure to adapt the output paths to your needs before running the script. Also check the comments in the file.

2. prepare_trainingfile.pl
This script prepares the initial training file. Outputs of step 1 are read and semantic classes are stored for each term. Since for most terms, there are multiple (conflicting) class assignments, the script outputs the majority class as the initial training class for classification. Note that these are the original ACL RD-TEC 2.0 classes before manual relabeling. Before running the script, make sure to adjust I/O paths.

The list of all terms in the corpus (including a major share of previously unlabeled data) was created mainly from the ACL RD-TEC 1.0 term list in combination extracted annotations from ACL RD-TEC 2.0. This is reflected in the output list in (folder output), which holds labels for more than 20,000 terms.