# complingterm
Comp Ling Term

This repository contains the materials for a coarse-grained classification of computational linguistics terms into semantic classes. The repository structure is as follows:

* data
	* training: Contains term lists and semantic classes used for training. Terms and classes were initially taken from the [ACL RD-TEC v. 1 and 2] (http://pars.ie/lr/acl_rd-tec). Some instances of the large "other" category were relabeled and classes were merged to larger coarse-grained classes. The resulting classes are:
		* DSMMM: data structures, mathematics, models, measurements
		* TechTool: technologies and tools
		* Linguistics: linguistic theories, linguistic units, language resources, language resource products
		* Remaining other
* src