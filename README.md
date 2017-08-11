# MCPixelCNN
MC-PixelCNN
Simple Version of [DeepSpace: Mood-based Image Texture Generation for Virtual Reality from Music](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Roy_DeepSpace_Mood-Based_Image_CVPR_2017_paper.pdf)

The data required should be Python cPickle file containing zip(image, phrase_emebddings, mood_embeddings)

Phrase Embeddings: Use GLoVe/ Word2Vec
Mood Embeddings: Either One hot encoding of moods/ Embeddings based on a trained Music-Mood Classifier. (We used the latter for the experiments)


