# complingterm

This repository contains the materials for a coarse-grained classification of computational linguistics terms from the ACL Anthology Reference Corpus (ACL ARC, http://acl-arc.comp.nus.edu.sg/) into semantic classes. The repository structure is as follows:

* data
	* training: Contains term lists and semantic classes used for training. Terms and classes were initially taken from the ACL RD-TEC v. 1 and 2 (http://pars.ie/lr/acl_rd-tec). Some instances of the large "other" category were relabeled and classes were merged to larger coarse-grained classes. The resulting classes are:
		* DSMMM: data structures, mathematics, models, measurements
		* TechTool: technologies and tools
		* Linguistics: linguistic theories, linguistic units, language resources, language resource products
		* Remaining other
	* output: The resulting term list with resulting classes, including training data. Most of these terms were previously *not annotated* in the ACL RD-TEC.
	* frequencies_over_time: Absolute and relative (normalised by the number of annotated terms per year) frequencies of semantic classes over all publication years. Our version of the ACL ARC spanned the years from 1965 up to 2006.
* src
	* data extraction: Tiny custom scripts for data preparation. Mainly put there for documentation.
	* classification: Python code for training and classification.

If you use the data or code provided here, please cite the [following paper](http://www.lrec-conf.org/proceedings/lrec2018/pdf/154.pdf) (where appropriate):

Anne-Kathrin Schumann, Héctor Martínez Alonso: "Automatic Annotation of Semantic Term Types in the Complete ACL Anthology Reference Corpus". 11th Language Resources and Evaluation Conference (LREC) 2018, pp. 3707-3715. Miyazaki, Japan, May 7-12, 2018. European Language Resources Association.
