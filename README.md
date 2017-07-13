# complingterm

This repository contains the materials for a coarse-grained classification of computational linguistics terms into semantic classes. The repository structure is as follows:

* data
	* training: Contains term lists and semantic classes used for training. Terms and classes were initially taken from the ACL RD-TEC v. 1 and 2 (http://pars.ie/lr/acl_rd-tec). Some instances of the large "other" category were relabeled and classes were merged to larger coarse-grained classes. The resulting classes are:
		* DSMMM: data structures, mathematics, models, measurements
		* TechTool: technologies and tools
		* Linguistics: linguistic theories, linguistic units, language resources, language resource products
		* Remaining other
	* output: The resulting term list with resulting classes. These terms were previously *not annotated* in the ACL RD-TEC.
	* frequencies_over_time: Absolute and relative (normalised by the number of annotated terms per year) frequencies of semantic classes over all publication years. Our version of the ACL ARC (http://acl-arc.comp.nus.edu.sg/) spanned the years from 1965 up to 2006.
* src: Python code for training and classification.
