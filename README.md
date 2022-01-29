<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Similar product recommendation using CNN</div>
<div align="center"><img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/intro.gif?raw=true"></div>




## Overview:
In this project, we will use a CNN Model to create a Fashion Embedding. This information can be used in ML algorithms with higher semantic quality and similarity between Objects. We will use embeddings to identify similar items, this information will be used to recommend similar content.

## What is Embedding?
An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.
So a natural language modelling technique like Word Embedding is used to map words or phrases from a vocabulary to a corresponding vector of real numbers. As well as being amenable to processing by learning algorithms, this vector representation has two important and advantageous properties:

- **Dimensionality Reduction** — it is a more efficient representation
- **Contextual Similarity** — it is a more expressive representation

<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/recommendation%20system%20using%20CNN.png?raw=true">

## Dataset:
[Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)
### Context: 
Thr growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon. In addition to professionally shot high resolution product images, we also have multiple label attributes describing the product which was manually entered while cataloging. To add to this, we also have descriptive text that comments on the product characteristics.

### Content:
Each product is identified by an ID like 42431. You will find a map to all the products in styles.csv. From here, you can fetch the image for this product from images/42431.jpg and the complete metadata from styles/42431.json.

To get started easily, we also have exposed some of the key product categories and it's display name in styles.csv.


## Implementation:

**libraries** : `matplotlib` `numpy` `pandas` `sklearn` `os` `keras` `tensorflow`
## Data Exploration:
The Dataset is made up of different items that can be found in a marketplace. The idea is to use embeddings to search for similarity and find similar items just using the image.

<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/eda1.PNG?raw=true" width="70%">
<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/eda2.PNG?raw=true" width="80%">

#### Top categories:

<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/eda3.PNG?raw=true">


## Pre-trained model for recommendation:
We'll use Resnet50.
```
img_ht, img_wt, _  = 224,224,3
base_model =  ResNet50(weights= "imagenet", include_top = False, input_shape =(img_wt, img_ht,3))
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

model.summary()
```
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
resnet50 (Functional)        (None, 7, 7, 2048)        23587712  
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 2048)              0         
=================================================================
Total params: 23,587,712
Trainable params: 0
Non-trainable params: 23,587,712
_________________________________________________________________
```

## Get embeddings for all the items in the dataset:
We'll use following methid for getting embeddings:
```
def embeddings(model, img_name):
    # Reshape
    img = image.load_img(img_path(img_name), target_size=(img_wt, img_ht))
    # img to Array
    x   = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)
```
## Compute Cosine similarity:
<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/cosine%20similarity.PNG?raw=true" width="65%">

<br>
```
# Calculating pairwise similarity
from sklearn.metrics.pairwise import pairwise_distances

#cosine distances
cosine_similarity = 1- pairwise_distances(df_embs, metric = "cosine")
cosine_similarity[:4, :4]
```
```
array([[0.99999964, 0.63589436, 0.49575073, 0.59518987],
       [0.63589436, 1.        , 0.509784  , 0.7375228 ],
       [0.49575073, 0.509784  , 1.        , 0.471317  ],
       [0.59518987, 0.7375228 , 0.471317  , 0.9999991 ]], dtype=float32)
```
## Get recommendations:
```
def get_recommendations(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_similarity[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim

get_recommendations(2993, df, top_n = 5)
```
#### Image:
<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/image.PNG?raw=true" width="50%">

#### Recommendations:
<img src="https://github.com/Pradnya1208/Similar-product-recommendation-system-using-CNN/blob/main/images/reco.PNG?raw=true">

<br>

Convolutional networks can be used to generate generic embeddings of any content. These embeddings can be used to identify similar items and in a recommendation process.




### Learnings:
`Recommendation systems` 






## References:
[CNN based recommendation](https://www.hindawi.com/journals/complexity/2019/9476981/)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner




[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]
