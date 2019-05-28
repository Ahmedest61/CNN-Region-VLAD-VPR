# A Holistic Visual Place Recognition Approach using Lightweight CNNs for Severe ViewPoint and Appearance Changes

- There are five benchmark datasets tested on the proposed methodology:
1) Berlin Halenseestrasse
2) Berlin Kudamm
3) Berlin A100
4) Garden Point
5) Syhthesized Nordland

If you use these datasets, please cite the following publication:

```
@article{khaliq2018holistic,
  title={A Holistic Visual Place Recognition Approach using Lightweight CNNs for Severe ViewPoint and Appearance Changes},
  author={Khaliq, Ahmad and Ehsan, Shoaib and Milford, Michael and McDonald-Maier, Klaus},
  journal={arXiv preprint arXiv:1811.03032},
  year={2018}
}
```

- Each dataset contains two traverses of the same route under different viewpoints and conditions
	- A "Result" folder in each dataset contains the matched and unmatched image files using the proposed approach
		- For each test image, the retrived image is correct if its name is written in green color else red color shows it is treated as unmatched
		- In some unmatched scenarios, we observed that the retrieved image is quite similar/closer to the test image but due to the ground truth priorities, our algorithm kept those unmatched which reflects in the AUC-PR curve.
	- Pickle files are there for two settings i.e. N= 200 Regions with V =128 clusters and N =400 Regions with V =256 clusters 	
		- Each file is a dictionary containing keys as test images names and values contain the retrieval information i.e. scores , retrieved image names etc
			- Each Value is a list which contains the name of ground truth image, predicted label, prediction score and the retrieved image name

- There is another folder "Vocabulary", containing dataset "2.6K", employed for making regional dictionaries
	- Two pickle files are there, one with N= 400 regions clustered into V= 64,128,256 regions, where the other file contains N= 100,200,300 regions clustered again into V= 64,128,256 regions each. Each file is again a nested dictionary with nested keys as Region (N) and  Cluster (V). 

- A python script "produceResults.py" can generate the results i.e. AUC-PR and retrieved images for both the configurations using the Pickle files. The user just need to defined the "datasetIndex" and "dir" parameter.

Configuration 1: N = 200, V=128
 	(AUC-PR Results)
1) Berlin Halenseestrasse: 0.754
2) Berlin Kudamm: 0.298
3) Berlin A100: 0.7381
4) Garden Point: 0.6540
5) Synthesized Nordland: 0.539

Configuration 2: N = 400, V=256
	(AUC-PR Results)
1) Berlin Halenseestrasse: 0.808
2) Berlin Kudamm: 0.395
3) Berlin A100: 0.7117
4) Garden Point: 0.7225
5) Synthesized Nordland: 0.547

