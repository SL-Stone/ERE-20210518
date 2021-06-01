The dataset  NYT is released by [Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism](https://www.aclweb.org/anthology/P18-1047.pdf).

It contains 56195 sentences for training, 5000 sentences for validation(but only 4999 senences when I program), and 5000 sentences for test.

First download CopyR's NYT data from https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3 to the dir raw_NYT/

Then process the data at the dir raw_NYT/

Finally run build_data.py to get triple files.

I used the method in [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://arxiv.org/abs/1909.03227) to preprocess the NYT data. Many thanks to them.
