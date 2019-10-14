# [The 3rd YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m-2019)
## Temporal localization of topics within video
### Google Research x Kaggle

![](http://research.google.com/youtube8m/workshop2018/logo.png)

----

Official Kaggle Competition: https://www.kaggle.com/c/youtube8m-2019

<br>

This repo contains the **13th place solution** for training and evaluating machine learning models over the **YouTube-8M dataset**. This is our solution for 3rd Youtube8M Video Understanding Challenge on Kaggle and presented at the **International Conference on Computer Vision (ICCV)** 2019. The code gives an end-to-end working example for reading the dataset, training a TensorFlow model, and evaluating the performance of the model.

Please check the official github repository: https://github.com/google/youtube-8m and the [Starter Code](https://github.com/google/youtube-8m)

---

## About the team

Team name: **Team Locust**

**Members:**
- [LijunZhang](https://www.kaggle.com/lijunleozhang)
- [Srinath Nizampatnam](https://www.kaggle.com/srinath1993)
- [Ahana Gangopadhyay](https://www.kaggle.com/ahana91)
- [Nanashi (you)](https://www.kaggle.com/jesucristo)

**Leaderboard**

<a href="https://www.kaggle.com/c/youtube8m-2019/leaderboard"><img src="https://i.ibb.co/WkNwnww/Screenshot-from-2019-10-14-23-13-08.png" alt="Screenshot-from-2019-10-14-23-13-08" border="0"></a>

---

## [13th place solution summary](https://www.kaggle.com/c/youtube8m-2019/discussion/112298#latest-647146)

**Standard frame-level models**
We first tried out the models from 1st and 2nd competition wining solutions, including different variations of netVlad, DBoF and LSTM. With the standard training on regular training set, the best result is similar to frame-level DBoF baseline model in the leaderboard (I think we joined the competition mid-July and it took us one month to achieve this. Before that we were stuck into a silly problem. We always used Excel to open the large predictions.csv file. But somehow Excel will automatically modify the large cell and make the submission failed. It let us feels like I should keep segmentmaxpred, which is total number of segment outputs per entity, very low. Anyway, we figured it out after long time).

**Fine-tuning on validation set with segment labels**

The next thing we tried out and worked is fine-tuning on the labelled segments on the validation dataset. It will generally give us aprox 0.04 improvement. Our training goes into two phases: Phase 1, we trained the model on 1.4 TB regular training set. Phase 2, we fine-tuned the model on the validation set with segment labels. There are usually around 60K training steps in phase 1 and we only did around 2000~3000 steps of fine-tuning.
We found that more steps of fine-tuning lead to worst results. Probably it is because more fine-tuning steps will lead the model to overfit validation dataset, whose size is comparable much smaller to regular training dataset.

**Attention/Multi-attention model**

In the regular training set, we have frames across whole video. In the validation set and testing set, we have 5-frames segments. To bridge the gap between regular training set and validation set, we decided to formulate the problem as a multi-instance learning problem and use the attention weights to select important frames in the video. In our model, each frame will go through an attention network to obtain the attention weight of that frame. Then we pooled the frames based on the attention weights and used MoE/logistic model to obtain the final prediction. We also use gating mechanism in the attention network as shown in the paper(https://arxiv.org/pdf/1802.04712.pdf).
As our problem is a multi-class classification problem, we also considered that different high-level topics may require different attention network to learn the weights. We also trained multiple (8 or 16 in final submission) attention networks to emphasize the different important frames in different topics of videos. Finally, we pooled the output from each of attention network in our model. Our best individual model in public LB is 0.772.

**Loss function and final ensemble**
As in the final testing set, the segments only come from 1000 classes. In the loss function, we gave those 1000 classes more weights. Our final ensemble consists of 3 multi-attention models, 3 attention models, 2 DBoF models, 2 CNN models, 1 netVlad model and 1 LSTM model.

**What was tried but didn’t work**

We put a lot of effort on data augmentation and semi-supervised learning. In the beginning, we tried to use our best model to predict segments in the regular training dataset and choose top segments as our new training set. We also tried to pseudo-label the testing dataset. We chose top 200 segments from our best submission and fine-tuned the model based on that. Both methods didn’t work out. One possible reason is we did not blend them with standard validation dataset during the fine-tuning. How we label these segments and how we use them require more experiments.
