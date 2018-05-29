# DrugAI
Classification of Drug Like molecule using Neural Networks.

more about DrugAI..
[http://gananath.github.io/drugai.html](http://gananath.github.io/drugai.html)
# Requirments
- Python 2.7

- Keras(Theano/Tensorflow)

- Pandas

- Scikit-Learn

# DrugAI-Gen.py (LSTM model)
Generator script for creating drug like molecule using LSTM model. 
Read more from here [http://gananath.github.io/drugai-gen.html](http://gananath.github.io/drugai-gen.html)

# DrugAI-GAN.py (GAN model)
This is my own experiments with Generative Adverserial Network (GAN) for drug like molecule generation. Teaching GAN in discrete dataset is hard and also I learned to code GAN from internet so would not gurantee any acurracy of the results or the code. 
Read more from here [http://gananath.github.io/drugai-gan.html](http://gananath.github.io/drugai-gan.html)

# DrugAI-WGAN.py (WassersteinGAN model)
A [Wasserstein GAN model](http://gananath.github.io/drugai-gan.html) with CNN; this model currently trains the fastest and probably gives the best result.

```
# Samples Generated 
['CC1=C(C(C(=O)O)(=CC=N2[S]CCCCCC(C(Cl)C1C4)[+])C2=C4=O|||||||||||||||||||||||||||' 'CC1=C(C(C(=O)OO(=CC=N2[N]CCCCCC(C(Cl)C1C3)[+])C2=CC=O)||||||||||||||||||||||||||' 'CC1=C(C(C(=O)O)(=CC=N2[N]CCC=CC(C(CO)C1C3)[+])C2=CC=O)||||||||||||||||||||||||||']
```
- Recently I came across [GAN's which uses condition](https://camo.githubusercontent.com/df22e45e90834484356be762450ffc5f66c34a83/68747470733a2f2f7062732e7477696d672e636f6d2f6d656469612f43774d30427a6a5655414157546e342e6a70673a6c61726765) like cGAN,acGAN etc. which uses a **noise+class** as input whereas I am using only **class**.

# Another Dataset for Generation
Because I seen a increase in interest for **DrugAI-Gen.py**; for programmers I have added another dataset *sms.tsv*. It contains SMS spams. Try to use it for generating *Spam's* and *Ham's*.

# citation
```
@misc{gananath2016,
  author = {Gananath, R.},
  title = {DrugAI},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Gananath/DrugAI}}
}
```
