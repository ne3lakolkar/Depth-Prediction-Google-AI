![](https://ad.doubleclick.net/ddm/activity/src=2542116;type=gblog;cat=googl0;ord=1?)

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/GoogleAI_logo_horizontal_color_rgb.png)](http://ai.googleblog.com/) [](https://ai.googleblog.com/)

Blog
====

The latest news from Google AI

[Moving Camera, Moving People: A Deep Learning Approach to Depth Prediction](http://ai.googleblog.com/2019/05/moving-camera-moving-people-deep.html "Moving Camera, Moving People: A Deep Learning Approach to Depth Prediction")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Thursday, May 23, 2019

Posted by Tali Dekel, Research Scientist and Forrester Cole, Software Engineer, Machine Perception
 The human visual system has a remarkable ability to make sense of our 3D world from its 2D projection. Even in complex environments with multiple moving objects, people are able to maintain a feasible interpretation of the objects’ geometry and depth ordering. The field of [computer vision](https://en.wikipedia.org/wiki/Computer_vision) has long studied how to achieve similar capabilities by computationally reconstructing a scene’s geometry from 2D image data, but robust reconstruction remains difficult in many cases.
 A particularly challenging case occurs when both the camera and the objects in the scene are freely moving. This confuses traditional 3D reconstruction algorithms that are based on [triangulation](https://en.wikipedia.org/wiki/Triangulation_(computer_vision)), which assumes that the same object can be observed from at least two different viewpoints, at the same time. Satisfying this assumption requires either a multi-camera array (like [Google’s Jump](https://vr.google.com/jump/)), or a scene that remains stationary as the single camera moves through it. As a result, most existing methods either filter out moving objects (assigning them “zero” depth values), or ignore them (resulting in incorrect depth values).

||
|[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/image7.png)](https://3.bp.blogspot.com/-vsChoOi9kqg/XORkXqQFtpI/AAAAAAAAEIM/LkBVYsz6RmUeMPukh-R1BKoaqchGh9ryQCLcBGAs/s1600/image7.png)|
|**Left:** The traditional stereo setup assumes that at least two viewpoints capture the scene at the same time. **Right:** We consider the setup where both camera and subject are moving.|

In “[Learning the Depths of Moving People by Watching Frozen People](https://arxiv.org/pdf/1904.11111.pdf)”, we tackle this fundamental challenge by applying a [deep learning-based](https://en.wikipedia.org/wiki/Deep_learning) approach that can generate [depth maps](https://en.wikipedia.org/wiki/Depth_map) from an ordinary video, where both the camera and subjects are freely moving. The model avoids direct 3D triangulation by learning priors on human pose and shape from data. While there is a recent surge in using machine learning for depth prediction, this work is the first to tailor a learning-based approach to the case of simultaneous camera and human motion. In this work, we focus specifically on humans because they are an interesting target for [augmented reality](https://en.wikipedia.org/wiki/Augmented_reality) and 3D video effects.

||
|[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/image4.gif)](https://2.bp.blogspot.com/-CotMQ8DsT-Y/XORkuhmLQtI/AAAAAAAAEIk/VfzxNYJFueQ0QzjLCbSIcOljDZLynexoACLcBGAs/s1600/image4.gif)|
|Our model predicts the depth map (**right**; brighter=closer to the camera) from a [regular video](https://www.shutterstock.com/video/clip-26820115-handsome-young-businessman-doing-victory-dance-about) (**left**), where both the people in the scene and the camera are freely moving.|

**Sourcing the Training Data**
 We train our depth-prediction model in a supervised manner, which requires videos of natural scenes, captured by moving cameras, along with accurate depth maps. The key question is where to get such data. Generating data synthetically requires realistic modeling and rendering of a wide range of scenes and natural human actions, which is challenging. Further, a model trained on such data may have difficulty generalizing to real scenes. Another approach might be to record real scenes with an RGBD sensor (e.g., Microsoft’s Kinect), but depth sensors are typically limited to indoor environments and have their own set of 3D reconstruction issues.
 Instead, we make use of an existing source of data for supervision: YouTube videos in which [people imitate mannequins](https://en.wikipedia.org/wiki/Mannequin_Challenge) by freezing in a wide variety of natural poses, while a hand-held camera tours the scene. Because the entire scene is stationary (only the camera is moving), triangulation-based methods--like [multi-view-stereo](https://en.wikipedia.org/wiki/3D_reconstruction_from_multiple_images) (MVS)--work, and we can get accurate depth maps for the entire scene including the people in it. We gathered approximately 2000 such videos, spanning a wide range of realistic scenes with people naturally posing in different group configurations.

||
|[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/dataset_compressed.gif)](https://1.bp.blogspot.com/-EtavC4Ux6Eg/XOWy6l6kV7I/AAAAAAAAEJ4/2g8VZIRS51o95R2BwZ-dgL79uRA_ogDXQCLcBGAs/s1600/dataset_compressed.gif)|
|Videos of [people imitating mannequins](https://en.wikipedia.org/wiki/Mannequin_Challenge) while a camera tours the scene, which we used for training. We use traditional MVS algorithms to estimate depth, which serves as supervision during training of our depth-prediction model.|

**Inferring the Depth of Moving People**
 The Mannequin Challenge videos provide depth supervision for moving camera and “frozen” people, but our goal is to handle videos with a moving camera *and moving people*. We need to structure the input to the network in order to bridge that gap.
 A possible approach is to infer depth separately for each frame of the video (i.e., the input to the model is just a single frame). While such a model already improves over state-of-the-art single image methods for depth prediction, we can improve the results further by considering information from multiple frames. For example, [motion parallax](https://en.wikipedia.org/wiki/Parallax), i.e., the relative apparent motion of *static objects* between two different viewpoints, provides strong depth cues. To benefit from such information, we compute the [2D optical flow](https://en.wikipedia.org/wiki/Optical_flow) between each input frame and another frame in the video, which represents the pixel displacement between the two frames. This flow field depends on both the scene’s depth and the relative position of the camera. However, because the camera positions are known, we can remove their dependency from the flow field, which results in an initial depth map. This initial depth is valid only for static scene regions. To handle moving people at test time, we apply a [human-segmentation network](https://arxiv.org/abs/1703.06870) to mask out human regions in the initial depth map. The full input to our network then includes: the RGB image, the human mask, and the masked depth map from parallax.

||
|[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/image5.png)](https://3.bp.blogspot.com/-UdkbY38FlnE/XORkuolgt1I/AAAAAAAAEI4/EhyqCrN7c2okJPkOvxODT2RE4fLKUPPNACEwYBhgL/s1600/image5.png)|
|Depth prediction network: The input to the model includes an RGB image (Frame *t*), a mask of the human region, and an initial depth for the non-human regions, computed from motion parallax (optical flow) between the input frame and another frame in the video. The model outputs a full depth map for Frame *t*. Supervision for training is provided by the depth map, computed by MVS.|

The network’s job is to “inpaint” the depth values for the regions with people, and refine the depth elsewhere. Intuitively, because humans have consistent shape and physical dimensions, the network can internally learn such priors by observing many training examples. Once trained, our model can handle natural videos with arbitrary camera and human motion.

Below are some examples of our depth-prediction model results based on videos, with comparison to recent state-of-the-art learning based methods.

||
|[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/image1.gif)](https://4.bp.blogspot.com/-do3CDsS64Wk/XOWB1t-Or1I/AAAAAAAAEJg/jr8Jg1B14OQUmebDrGybl9bn-zvMpk7AwCLcBGAs/s1600/image1.gif)|
|Comparison of depth prediction models to a [video clip](https://www.shutterstock.com/video/clip-11468558-asian-sisters-running-around-park-laughing-together) with moving cameras and people. **Top:** Learning based monocular depth prediction methods ([DORN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.pdf); [Chen et al.](https://papers.nips.cc/paper/6489-single-image-depth-perception-in-the-wild.pdf)). **Bottom:** Learning based stereo method ([DeMoN](https://www.google.com/url?q=http://openaccess.thecvf.com/content_cvpr_2017/papers/Ummenhofer_DeMoN_Depth_and_CVPR_2017_paper.pdf&sa=D&ust=1558037062352000&usg=AFQjCNHoiGX0T8KkR9eTAxHQ3wAGK8pWgA)), and our result.|

**3D Video Effects Using Our Depth Maps**
 Our predicted depth maps can be used to produce a range of 3D-aware video effects. One such effect is synthetic defocus. Below is an example, produced from an ordinary video using our depth map.

||
|[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/vid4_compressed.gif)](https://1.bp.blogspot.com/-6J23_0pC50U/XOV0XjNrdwI/AAAAAAAAEJE/VTShoEITktgcGLhSEz3QOL_DiGs9Ae0tQCLcBGAs/s1600/vid4_compressed.gif)|
|Bokeh video effect produced using our estimated depth maps. Video courtesy of [Wind Walk Travel Videos](https://www.youtube.com/channel/UCPur06mx78RtwgHJzxpu2ew).|

Other possible applications for our depth maps include generating a stereo video from a monocular one, and inserting synthetic CG objects into the scene. Depth maps also provide the ability to fill in holes and disoccluded regions with the content exposed in other frames of the video. In the following example, we have synthetically wiggled the camera at several frames and filled in the regions behind the actor with pixels from other frames of the video.

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/image8.gif)](https://3.bp.blogspot.com/-RYv0ZbQr42c/XORkwPL268I/AAAAAAAAEI8/FgGdlKtHikYDfeJcgb_LYpW2CzxVSEiswCEwYBhgL/s1600/image8.gif)

**Acknowledgements**
 *The research described in this post was done by Zhengqi Li, Tali Dekel, Forrester Cole, Richard Tucker, Noah Snavely, Ce Liu and Bill Freeman. We would like to thank Miki Rubinstein for his valuable feedback.*

![Share on Twitter](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/post_twitter_black_24dp.png) ![Share on Facebook](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/post_facebook_black_24dp.png)

[**](http://ai.googleblog.com/) [**](http://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html "Newer Post") [**](http://ai.googleblog.com/2019/05/introducing-translatotron-end-to-end.html "Older Post")

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/icon18_wrench_allbkg.png)](https://www.blogger.com/rearrange?blogID=8474926331452026626&widgetType=HTML&widgetId=HTML8&action=editWidget&sectionId=sidebar-top "Edit")

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAYpJREFUeNrs2aFuwzAQBmAvKRkMKRjZA4QMDJaWFgyMjuzFRg37DIUlA3uFkoGQSaWzJU+tpri5O9+l/zSfdFJlpe59yTmyVedq1PjfcZMZ70NuQnaF8w8htyE/rABtpviXkLcK88c5HhLkMBfgVan43zfFBNGMjHVGT/s55KP2pAvidbGHd+nzKt1RKSLG3rKF1iPFv6UWiPke8i7kEqGdGsI1O+LYVdqJAjgirwkKYD0ytkJBUNbAMvX8V3q9PhUsYvU1sWD8SO/sQvx2ahxOiNoJCSBCoAHYCEQAC4EKICOQASQEOmAS8RcAFxFN5hiIiugpgC3wk9hQAHH/70EBHXUN7IER5EWMiBgo2+nzOKQv9SCAeEM/OQAkhE/ncccFICB87qzQMia5FsJfOui0zMnmRvipU1ormHQuxGTxUsAcCFLxJQBLBLn4UoAFglW8BkATwS5eC6CBEBWvCShBiIvXBkgQRcVbADiI4uKtABSESvGWgB9EzHt3+tNwyO0qa9SoIYtvAQYAqDJhaWWeMecAAAAASUVORK5CYII=)

Labels
------

**

-   [accessibility](http://ai.googleblog.com/search/label/accessibility)
-   [ACL](http://ai.googleblog.com/search/label/ACL)
-   [ACM](http://ai.googleblog.com/search/label/ACM)
-   [Acoustic Modeling](http://ai.googleblog.com/search/label/Acoustic%20Modeling)
-   [Adaptive Data Analysis](http://ai.googleblog.com/search/label/Adaptive%20Data%20Analysis)
-   [ads](http://ai.googleblog.com/search/label/ads)
-   [adsense](http://ai.googleblog.com/search/label/adsense)
-   [adwords](http://ai.googleblog.com/search/label/adwords)
-   [Africa](http://ai.googleblog.com/search/label/Africa)
-   [AI](http://ai.googleblog.com/search/label/AI)
-   [AI for Social Good](http://ai.googleblog.com/search/label/AI%20for%20Social%20Good)
-   [Algorithms](http://ai.googleblog.com/search/label/Algorithms)
-   [Android](http://ai.googleblog.com/search/label/Android)
-   [Android Wear](http://ai.googleblog.com/search/label/Android%20Wear)
-   [API](http://ai.googleblog.com/search/label/API)
-   [App Engine](http://ai.googleblog.com/search/label/App%20Engine)
-   [App Inventor](http://ai.googleblog.com/search/label/App%20Inventor)
-   [April Fools](http://ai.googleblog.com/search/label/April%20Fools)
-   [Art](http://ai.googleblog.com/search/label/Art)
-   [Audio](http://ai.googleblog.com/search/label/Audio)
-   [Augmented Reality](http://ai.googleblog.com/search/label/Augmented%20Reality)
-   [Australia](http://ai.googleblog.com/search/label/Australia)
-   [Automatic Speech Recognition](http://ai.googleblog.com/search/label/Automatic%20Speech%20Recognition)
-   [AutoML](http://ai.googleblog.com/search/label/AutoML)
-   [Awards](http://ai.googleblog.com/search/label/Awards)
-   [BigQuery](http://ai.googleblog.com/search/label/BigQuery)
-   [Cantonese](http://ai.googleblog.com/search/label/Cantonese)
-   [Chemistry](http://ai.googleblog.com/search/label/Chemistry)
-   [China](http://ai.googleblog.com/search/label/China)
-   [Chrome](http://ai.googleblog.com/search/label/Chrome)
-   [Cloud Computing](http://ai.googleblog.com/search/label/Cloud%20Computing)
-   [Collaboration](http://ai.googleblog.com/search/label/Collaboration)
-   [Compression](http://ai.googleblog.com/search/label/Compression)
-   [Computational Imaging](http://ai.googleblog.com/search/label/Computational%20Imaging)
-   [Computational Photography](http://ai.googleblog.com/search/label/Computational%20Photography)
-   [Computer Science](http://ai.googleblog.com/search/label/Computer%20Science)
-   [Computer Vision](http://ai.googleblog.com/search/label/Computer%20Vision)
-   [conference](http://ai.googleblog.com/search/label/conference)
-   [conferences](http://ai.googleblog.com/search/label/conferences)
-   [Conservation](http://ai.googleblog.com/search/label/Conservation)
-   [correlate](http://ai.googleblog.com/search/label/correlate)
-   [Course Builder](http://ai.googleblog.com/search/label/Course%20Builder)
-   [crowd-sourcing](http://ai.googleblog.com/search/label/crowd-sourcing)
-   [CVPR](http://ai.googleblog.com/search/label/CVPR)
-   [Data Center](http://ai.googleblog.com/search/label/Data%20Center)
-   [Data Discovery](http://ai.googleblog.com/search/label/Data%20Discovery)
-   [data science](http://ai.googleblog.com/search/label/data%20science)
-   [datasets](http://ai.googleblog.com/search/label/datasets)
-   [Deep Learning](http://ai.googleblog.com/search/label/Deep%20Learning)
-   [DeepDream](http://ai.googleblog.com/search/label/DeepDream)
-   [DeepMind](http://ai.googleblog.com/search/label/DeepMind)
-   [distributed systems](http://ai.googleblog.com/search/label/distributed%20systems)
-   [Diversity](http://ai.googleblog.com/search/label/Diversity)
-   [Earth Engine](http://ai.googleblog.com/search/label/Earth%20Engine)
-   [economics](http://ai.googleblog.com/search/label/economics)
-   [Education](http://ai.googleblog.com/search/label/Education)
-   [Electronic Commerce and Algorithms](http://ai.googleblog.com/search/label/Electronic%20Commerce%20and%20Algorithms)
-   [electronics](http://ai.googleblog.com/search/label/electronics)
-   [EMEA](http://ai.googleblog.com/search/label/EMEA)
-   [EMNLP](http://ai.googleblog.com/search/label/EMNLP)
-   [Encryption](http://ai.googleblog.com/search/label/Encryption)
-   [entities](http://ai.googleblog.com/search/label/entities)
-   [Entity Salience](http://ai.googleblog.com/search/label/Entity%20Salience)
-   [Environment](http://ai.googleblog.com/search/label/Environment)
-   [Europe](http://ai.googleblog.com/search/label/Europe)
-   [Exacycle](http://ai.googleblog.com/search/label/Exacycle)
-   [Expander](http://ai.googleblog.com/search/label/Expander)
-   [Faculty Institute](http://ai.googleblog.com/search/label/Faculty%20Institute)
-   [Faculty Summit](http://ai.googleblog.com/search/label/Faculty%20Summit)
-   [Flu Trends](http://ai.googleblog.com/search/label/Flu%20Trends)
-   [Fusion Tables](http://ai.googleblog.com/search/label/Fusion%20Tables)
-   [gamification](http://ai.googleblog.com/search/label/gamification)
-   [Gboard](http://ai.googleblog.com/search/label/Gboard)
-   [Gmail](http://ai.googleblog.com/search/label/Gmail)
-   [Google Accelerated Science](http://ai.googleblog.com/search/label/Google%20Accelerated%20Science)
-   [Google Books](http://ai.googleblog.com/search/label/Google%20Books)
-   [Google Brain](http://ai.googleblog.com/search/label/Google%20Brain)
-   [Google Cloud Platform](http://ai.googleblog.com/search/label/Google%20Cloud%20Platform)
-   [Google Docs](http://ai.googleblog.com/search/label/Google%20Docs)
-   [Google Drive](http://ai.googleblog.com/search/label/Google%20Drive)
-   [Google Genomics](http://ai.googleblog.com/search/label/Google%20Genomics)
-   [Google Maps](http://ai.googleblog.com/search/label/Google%20Maps)
-   [Google Photos](http://ai.googleblog.com/search/label/Google%20Photos)
-   [Google Play Apps](http://ai.googleblog.com/search/label/Google%20Play%20Apps)
-   [Google Science Fair](http://ai.googleblog.com/search/label/Google%20Science%20Fair)
-   [Google Sheets](http://ai.googleblog.com/search/label/Google%20Sheets)
-   [Google Translate](http://ai.googleblog.com/search/label/Google%20Translate)
-   [Google Trips](http://ai.googleblog.com/search/label/Google%20Trips)
-   [Google Voice Search](http://ai.googleblog.com/search/label/Google%20Voice%20Search)
-   [Google+](http://ai.googleblog.com/search/label/Google%2B)
-   [Government](http://ai.googleblog.com/search/label/Government)
-   [grants](http://ai.googleblog.com/search/label/grants)
-   [Graph](http://ai.googleblog.com/search/label/Graph)
-   [Graph Mining](http://ai.googleblog.com/search/label/Graph%20Mining)
-   [Hardware](http://ai.googleblog.com/search/label/Hardware)
-   [HCI](http://ai.googleblog.com/search/label/HCI)
-   [Health](http://ai.googleblog.com/search/label/Health)
-   [High Dynamic Range Imaging](http://ai.googleblog.com/search/label/High%20Dynamic%20Range%20Imaging)
-   [ICCV](http://ai.googleblog.com/search/label/ICCV)
-   [ICLR](http://ai.googleblog.com/search/label/ICLR)
-   [ICML](http://ai.googleblog.com/search/label/ICML)
-   [ICSE](http://ai.googleblog.com/search/label/ICSE)
-   [Image Annotation](http://ai.googleblog.com/search/label/Image%20Annotation)
-   [Image Classification](http://ai.googleblog.com/search/label/Image%20Classification)
-   [Image Processing](http://ai.googleblog.com/search/label/Image%20Processing)
-   [Inbox](http://ai.googleblog.com/search/label/Inbox)
-   [India](http://ai.googleblog.com/search/label/India)
-   [Information Retrieval](http://ai.googleblog.com/search/label/Information%20Retrieval)
-   [internationalization](http://ai.googleblog.com/search/label/internationalization)
-   [Internet of Things](http://ai.googleblog.com/search/label/Internet%20of%20Things)
-   [Interspeech](http://ai.googleblog.com/search/label/Interspeech)
-   [IPython](http://ai.googleblog.com/search/label/IPython)
-   [Journalism](http://ai.googleblog.com/search/label/Journalism)
-   [jsm](http://ai.googleblog.com/search/label/jsm)
-   [jsm2011](http://ai.googleblog.com/search/label/jsm2011)
-   [K-12](http://ai.googleblog.com/search/label/K-12)
-   [Kaggle](http://ai.googleblog.com/search/label/Kaggle)
-   [KDD](http://ai.googleblog.com/search/label/KDD)
-   [Keyboard Input](http://ai.googleblog.com/search/label/Keyboard%20Input)
-   [Klingon](http://ai.googleblog.com/search/label/Klingon)
-   [Korean](http://ai.googleblog.com/search/label/Korean)
-   [Labs](http://ai.googleblog.com/search/label/Labs)
-   [Linear Optimization](http://ai.googleblog.com/search/label/Linear%20Optimization)
-   [localization](http://ai.googleblog.com/search/label/localization)
-   [Low-Light Photography](http://ai.googleblog.com/search/label/Low-Light%20Photography)
-   [Machine Hearing](http://ai.googleblog.com/search/label/Machine%20Hearing)
-   [Machine Intelligence](http://ai.googleblog.com/search/label/Machine%20Intelligence)
-   [Machine Learning](http://ai.googleblog.com/search/label/Machine%20Learning)
-   [Machine Perception](http://ai.googleblog.com/search/label/Machine%20Perception)
-   [Machine Translation](http://ai.googleblog.com/search/label/Machine%20Translation)
-   [Magenta](http://ai.googleblog.com/search/label/Magenta)
-   [MapReduce](http://ai.googleblog.com/search/label/MapReduce)
-   [market algorithms](http://ai.googleblog.com/search/label/market%20algorithms)
-   [Market Research](http://ai.googleblog.com/search/label/Market%20Research)
-   [Mixed Reality](http://ai.googleblog.com/search/label/Mixed%20Reality)
-   [ML](http://ai.googleblog.com/search/label/ML)
-   [ML Fairness](http://ai.googleblog.com/search/label/ML%20Fairness)
-   [MOOC](http://ai.googleblog.com/search/label/MOOC)
-   [Moore's Law](http://ai.googleblog.com/search/label/Moore%27s%20Law)
-   [Multimodal Learning](http://ai.googleblog.com/search/label/Multimodal%20Learning)
-   [NAACL](http://ai.googleblog.com/search/label/NAACL)
-   [Natural Language Processing](http://ai.googleblog.com/search/label/Natural%20Language%20Processing)
-   [Natural Language Understanding](http://ai.googleblog.com/search/label/Natural%20Language%20Understanding)
-   [Network Management](http://ai.googleblog.com/search/label/Network%20Management)
-   [Networks](http://ai.googleblog.com/search/label/Networks)
-   [Neural Networks](http://ai.googleblog.com/search/label/Neural%20Networks)
-   [NeurIPS](http://ai.googleblog.com/search/label/NeurIPS)
-   [Nexus](http://ai.googleblog.com/search/label/Nexus)
-   [Ngram](http://ai.googleblog.com/search/label/Ngram)
-   [NIPS](http://ai.googleblog.com/search/label/NIPS)
-   [NLP](http://ai.googleblog.com/search/label/NLP)
-   [On-device Learning](http://ai.googleblog.com/search/label/On-device%20Learning)
-   [open source](http://ai.googleblog.com/search/label/open%20source)
-   [operating systems](http://ai.googleblog.com/search/label/operating%20systems)
-   [Optical Character Recognition](http://ai.googleblog.com/search/label/Optical%20Character%20Recognition)
-   [optimization](http://ai.googleblog.com/search/label/optimization)
-   [osdi](http://ai.googleblog.com/search/label/osdi)
-   [osdi10](http://ai.googleblog.com/search/label/osdi10)
-   [patents](http://ai.googleblog.com/search/label/patents)
-   [Peer Review](http://ai.googleblog.com/search/label/Peer%20Review)
-   [ph.d. fellowship](http://ai.googleblog.com/search/label/ph.d.%20fellowship)
-   [PhD Fellowship](http://ai.googleblog.com/search/label/PhD%20Fellowship)
-   [PhotoScan](http://ai.googleblog.com/search/label/PhotoScan)
-   [Physics](http://ai.googleblog.com/search/label/Physics)
-   [PiLab](http://ai.googleblog.com/search/label/PiLab)
-   [Pixel](http://ai.googleblog.com/search/label/Pixel)
-   [Policy](http://ai.googleblog.com/search/label/Policy)
-   [Professional Development](http://ai.googleblog.com/search/label/Professional%20Development)
-   [Proposals](http://ai.googleblog.com/search/label/Proposals)
-   [Public Data Explorer](http://ai.googleblog.com/search/label/Public%20Data%20Explorer)
-   [publication](http://ai.googleblog.com/search/label/publication)
-   [Publications](http://ai.googleblog.com/search/label/Publications)
-   [Quantum AI](http://ai.googleblog.com/search/label/Quantum%20AI)
-   [Quantum Computing](http://ai.googleblog.com/search/label/Quantum%20Computing)
-   [Recommender Systems](http://ai.googleblog.com/search/label/Recommender%20Systems)
-   [Reinforcement Learning](http://ai.googleblog.com/search/label/Reinforcement%20Learning)
-   [renewable energy](http://ai.googleblog.com/search/label/renewable%20energy)
-   [Research](http://ai.googleblog.com/search/label/Research)
-   [Research Awards](http://ai.googleblog.com/search/label/Research%20Awards)
-   [resource optimization](http://ai.googleblog.com/search/label/resource%20optimization)
-   [Robotics](http://ai.googleblog.com/search/label/Robotics)
-   [schema.org](http://ai.googleblog.com/search/label/schema.org)
-   [Search](http://ai.googleblog.com/search/label/Search)
-   [search ads](http://ai.googleblog.com/search/label/search%20ads)
-   [Security and Privacy](http://ai.googleblog.com/search/label/Security%20and%20Privacy)
-   [Self-Supervised Learning](http://ai.googleblog.com/search/label/Self-Supervised%20Learning)
-   [Semantic Models](http://ai.googleblog.com/search/label/Semantic%20Models)
-   [Semi-supervised Learning](http://ai.googleblog.com/search/label/Semi-supervised%20Learning)
-   [SIGCOMM](http://ai.googleblog.com/search/label/SIGCOMM)
-   [SIGMOD](http://ai.googleblog.com/search/label/SIGMOD)
-   [Site Reliability Engineering](http://ai.googleblog.com/search/label/Site%20Reliability%20Engineering)
-   [Social Networks](http://ai.googleblog.com/search/label/Social%20Networks)
-   [Software](http://ai.googleblog.com/search/label/Software)
-   [Sound Search](http://ai.googleblog.com/search/label/Sound%20Search)
-   [Speech](http://ai.googleblog.com/search/label/Speech)
-   [Speech Recognition](http://ai.googleblog.com/search/label/Speech%20Recognition)
-   [statistics](http://ai.googleblog.com/search/label/statistics)
-   [Structured Data](http://ai.googleblog.com/search/label/Structured%20Data)
-   [Style Transfer](http://ai.googleblog.com/search/label/Style%20Transfer)
-   [Supervised Learning](http://ai.googleblog.com/search/label/Supervised%20Learning)
-   [Systems](http://ai.googleblog.com/search/label/Systems)
-   [TensorBoard](http://ai.googleblog.com/search/label/TensorBoard)
-   [TensorFlow](http://ai.googleblog.com/search/label/TensorFlow)
-   [TPU](http://ai.googleblog.com/search/label/TPU)
-   [Translate](http://ai.googleblog.com/search/label/Translate)
-   [trends](http://ai.googleblog.com/search/label/trends)
-   [TTS](http://ai.googleblog.com/search/label/TTS)
-   [TV](http://ai.googleblog.com/search/label/TV)
-   [UI](http://ai.googleblog.com/search/label/UI)
-   [University Relations](http://ai.googleblog.com/search/label/University%20Relations)
-   [UNIX](http://ai.googleblog.com/search/label/UNIX)
-   [Unsupervised Learning](http://ai.googleblog.com/search/label/Unsupervised%20Learning)
-   [User Experience](http://ai.googleblog.com/search/label/User%20Experience)
-   [video](http://ai.googleblog.com/search/label/video)
-   [Video Analysis](http://ai.googleblog.com/search/label/Video%20Analysis)
-   [Virtual Reality](http://ai.googleblog.com/search/label/Virtual%20Reality)
-   [Vision Research](http://ai.googleblog.com/search/label/Vision%20Research)
-   [Visiting Faculty](http://ai.googleblog.com/search/label/Visiting%20Faculty)
-   [Visualization](http://ai.googleblog.com/search/label/Visualization)
-   [VLDB](http://ai.googleblog.com/search/label/VLDB)
-   [Voice Search](http://ai.googleblog.com/search/label/Voice%20Search)
-   [Wiki](http://ai.googleblog.com/search/label/Wiki)
-   [wikipedia](http://ai.googleblog.com/search/label/wikipedia)
-   [WWW](http://ai.googleblog.com/search/label/WWW)
-   [Year in Review](http://ai.googleblog.com/search/label/Year%20in%20Review)
-   [YouTube](http://ai.googleblog.com/search/label/YouTube)

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/icon18_wrench_allbkg.png)](https://www.blogger.com/rearrange?blogID=8474926331452026626&widgetType=Label&widgetId=Label1&action=editWidget&sectionId=sidebar "Edit")

**

Archive
-------

**

-   ** [**  ](javascript:void(0)) [2020](http://ai.googleblog.com/2020/)
    -   [Jan](http://ai.googleblog.com/2020/01/)

    -   [Feb](http://ai.googleblog.com/2020/02/)

    -   [Mar](http://ai.googleblog.com/2020/03/)

    -   [Apr](http://ai.googleblog.com/2020/04/)

    -   [May](http://ai.googleblog.com/2020/05/)

    -   [Jun](http://ai.googleblog.com/2020/06/)

    -   [Jul](http://ai.googleblog.com/2020/07/)

    -   [Aug](http://ai.googleblog.com/2020/08/)

-   ** [**  ](javascript:void(0)) [2019](http://ai.googleblog.com/2019/)
    -   [Jan](http://ai.googleblog.com/2019/01/)

    -   [Feb](http://ai.googleblog.com/2019/02/)

    -   [Mar](http://ai.googleblog.com/2019/03/)

    -   [Apr](http://ai.googleblog.com/2019/04/)

    -   [May](http://ai.googleblog.com/2019/05/)

    -   [Jun](http://ai.googleblog.com/2019/06/)

    -   [Jul](http://ai.googleblog.com/2019/07/)

    -   [Aug](http://ai.googleblog.com/2019/08/)

    -   [Sep](http://ai.googleblog.com/2019/09/)

    -   [Oct](http://ai.googleblog.com/2019/10/)

    -   [Nov](http://ai.googleblog.com/2019/11/)

    -   [Dec](http://ai.googleblog.com/2019/12/)

-   ** [**  ](javascript:void(0)) [2018](http://ai.googleblog.com/2018/)
    -   [Jan](http://ai.googleblog.com/2018/01/)

    -   [Feb](http://ai.googleblog.com/2018/02/)

    -   [Mar](http://ai.googleblog.com/2018/03/)

    -   [Apr](http://ai.googleblog.com/2018/04/)

    -   [May](http://ai.googleblog.com/2018/05/)

    -   [Jun](http://ai.googleblog.com/2018/06/)

    -   [Jul](http://ai.googleblog.com/2018/07/)

    -   [Aug](http://ai.googleblog.com/2018/08/)

    -   [Sep](http://ai.googleblog.com/2018/09/)

    -   [Oct](http://ai.googleblog.com/2018/10/)

    -   [Nov](http://ai.googleblog.com/2018/11/)

    -   [Dec](http://ai.googleblog.com/2018/12/)

-   ** [**  ](javascript:void(0)) [2017](http://ai.googleblog.com/2017/)
    -   [Jan](http://ai.googleblog.com/2017/01/)

    -   [Feb](http://ai.googleblog.com/2017/02/)

    -   [Mar](http://ai.googleblog.com/2017/03/)

    -   [Apr](http://ai.googleblog.com/2017/04/)

    -   [May](http://ai.googleblog.com/2017/05/)

    -   [Jun](http://ai.googleblog.com/2017/06/)

    -   [Jul](http://ai.googleblog.com/2017/07/)

    -   [Aug](http://ai.googleblog.com/2017/08/)

    -   [Sep](http://ai.googleblog.com/2017/09/)

    -   [Oct](http://ai.googleblog.com/2017/10/)

    -   [Nov](http://ai.googleblog.com/2017/11/)

    -   [Dec](http://ai.googleblog.com/2017/12/)

-   ** [**  ](javascript:void(0)) [2016](http://ai.googleblog.com/2016/)
    -   [Jan](http://ai.googleblog.com/2016/01/)

    -   [Feb](http://ai.googleblog.com/2016/02/)

    -   [Mar](http://ai.googleblog.com/2016/03/)

    -   [Apr](http://ai.googleblog.com/2016/04/)

    -   [May](http://ai.googleblog.com/2016/05/)

    -   [Jun](http://ai.googleblog.com/2016/06/)

    -   [Jul](http://ai.googleblog.com/2016/07/)

    -   [Aug](http://ai.googleblog.com/2016/08/)

    -   [Sep](http://ai.googleblog.com/2016/09/)

    -   [Oct](http://ai.googleblog.com/2016/10/)

    -   [Nov](http://ai.googleblog.com/2016/11/)

    -   [Dec](http://ai.googleblog.com/2016/12/)

-   ** [**  ](javascript:void(0)) [2015](http://ai.googleblog.com/2015/)
    -   [Jan](http://ai.googleblog.com/2015/01/)

    -   [Feb](http://ai.googleblog.com/2015/02/)

    -   [Mar](http://ai.googleblog.com/2015/03/)

    -   [Apr](http://ai.googleblog.com/2015/04/)

    -   [May](http://ai.googleblog.com/2015/05/)

    -   [Jun](http://ai.googleblog.com/2015/06/)

    -   [Jul](http://ai.googleblog.com/2015/07/)

    -   [Aug](http://ai.googleblog.com/2015/08/)

    -   [Sep](http://ai.googleblog.com/2015/09/)

    -   [Oct](http://ai.googleblog.com/2015/10/)

    -   [Nov](http://ai.googleblog.com/2015/11/)

    -   [Dec](http://ai.googleblog.com/2015/12/)

-   ** [**  ](javascript:void(0)) [2014](http://ai.googleblog.com/2014/)
    -   [Jan](http://ai.googleblog.com/2014/01/)

    -   [Feb](http://ai.googleblog.com/2014/02/)

    -   [Mar](http://ai.googleblog.com/2014/03/)

    -   [Apr](http://ai.googleblog.com/2014/04/)

    -   [May](http://ai.googleblog.com/2014/05/)

    -   [Jun](http://ai.googleblog.com/2014/06/)

    -   [Jul](http://ai.googleblog.com/2014/07/)

    -   [Aug](http://ai.googleblog.com/2014/08/)

    -   [Sep](http://ai.googleblog.com/2014/09/)

    -   [Oct](http://ai.googleblog.com/2014/10/)

    -   [Nov](http://ai.googleblog.com/2014/11/)

    -   [Dec](http://ai.googleblog.com/2014/12/)

-   ** [**  ](javascript:void(0)) [2013](http://ai.googleblog.com/2013/)
    -   [Jan](http://ai.googleblog.com/2013/01/)

    -   [Feb](http://ai.googleblog.com/2013/02/)

    -   [Mar](http://ai.googleblog.com/2013/03/)

    -   [Apr](http://ai.googleblog.com/2013/04/)

    -   [May](http://ai.googleblog.com/2013/05/)

    -   [Jun](http://ai.googleblog.com/2013/06/)

    -   [Jul](http://ai.googleblog.com/2013/07/)

    -   [Aug](http://ai.googleblog.com/2013/08/)

    -   [Sep](http://ai.googleblog.com/2013/09/)

    -   [Oct](http://ai.googleblog.com/2013/10/)

    -   [Nov](http://ai.googleblog.com/2013/11/)

    -   [Dec](http://ai.googleblog.com/2013/12/)

-   ** [**  ](javascript:void(0)) [2012](http://ai.googleblog.com/2012/)
    -   [Jan](http://ai.googleblog.com/2012/01/)

    -   [Feb](http://ai.googleblog.com/2012/02/)

    -   [Mar](http://ai.googleblog.com/2012/03/)

    -   [Apr](http://ai.googleblog.com/2012/04/)

    -   [May](http://ai.googleblog.com/2012/05/)

    -   [Jun](http://ai.googleblog.com/2012/06/)

    -   [Jul](http://ai.googleblog.com/2012/07/)

    -   [Aug](http://ai.googleblog.com/2012/08/)

    -   [Sep](http://ai.googleblog.com/2012/09/)

    -   [Oct](http://ai.googleblog.com/2012/10/)

    -   [Dec](http://ai.googleblog.com/2012/12/)

-   ** [**  ](javascript:void(0)) [2011](http://ai.googleblog.com/2011/)
    -   [Jan](http://ai.googleblog.com/2011/01/)

    -   [Feb](http://ai.googleblog.com/2011/02/)

    -   [Mar](http://ai.googleblog.com/2011/03/)

    -   [Apr](http://ai.googleblog.com/2011/04/)

    -   [May](http://ai.googleblog.com/2011/05/)

    -   [Jun](http://ai.googleblog.com/2011/06/)

    -   [Jul](http://ai.googleblog.com/2011/07/)

    -   [Aug](http://ai.googleblog.com/2011/08/)

    -   [Sep](http://ai.googleblog.com/2011/09/)

    -   [Nov](http://ai.googleblog.com/2011/11/)

    -   [Dec](http://ai.googleblog.com/2011/12/)

-   ** [**  ](javascript:void(0)) [2010](http://ai.googleblog.com/2010/)
    -   [Jan](http://ai.googleblog.com/2010/01/)

    -   [Feb](http://ai.googleblog.com/2010/02/)

    -   [Mar](http://ai.googleblog.com/2010/03/)

    -   [Apr](http://ai.googleblog.com/2010/04/)

    -   [May](http://ai.googleblog.com/2010/05/)

    -   [Jun](http://ai.googleblog.com/2010/06/)

    -   [Jul](http://ai.googleblog.com/2010/07/)

    -   [Aug](http://ai.googleblog.com/2010/08/)

    -   [Sep](http://ai.googleblog.com/2010/09/)

    -   [Oct](http://ai.googleblog.com/2010/10/)

    -   [Nov](http://ai.googleblog.com/2010/11/)

    -   [Dec](http://ai.googleblog.com/2010/12/)

-   ** [**  ](javascript:void(0)) [2009](http://ai.googleblog.com/2009/)
    -   [Jan](http://ai.googleblog.com/2009/01/)

    -   [Feb](http://ai.googleblog.com/2009/02/)

    -   [Mar](http://ai.googleblog.com/2009/03/)

    -   [Apr](http://ai.googleblog.com/2009/04/)

    -   [May](http://ai.googleblog.com/2009/05/)

    -   [Jun](http://ai.googleblog.com/2009/06/)

    -   [Jul](http://ai.googleblog.com/2009/07/)

    -   [Aug](http://ai.googleblog.com/2009/08/)

    -   [Nov](http://ai.googleblog.com/2009/11/)

    -   [Dec](http://ai.googleblog.com/2009/12/)

-   ** [**  ](javascript:void(0)) [2008](http://ai.googleblog.com/2008/)
    -   [Feb](http://ai.googleblog.com/2008/02/)

    -   [Mar](http://ai.googleblog.com/2008/03/)

    -   [Apr](http://ai.googleblog.com/2008/04/)

    -   [May](http://ai.googleblog.com/2008/05/)

    -   [Jul](http://ai.googleblog.com/2008/07/)

    -   [Sep](http://ai.googleblog.com/2008/09/)

    -   [Oct](http://ai.googleblog.com/2008/10/)

    -   [Nov](http://ai.googleblog.com/2008/11/)

    -   [Dec](http://ai.googleblog.com/2008/12/)

-   ** [**  ](javascript:void(0)) [2007](http://ai.googleblog.com/2007/)
    -   [Feb](http://ai.googleblog.com/2007/02/)

    -   [Jun](http://ai.googleblog.com/2007/06/)

    -   [Jul](http://ai.googleblog.com/2007/07/)

    -   [Aug](http://ai.googleblog.com/2007/08/)

    -   [Sep](http://ai.googleblog.com/2007/09/)

    -   [Oct](http://ai.googleblog.com/2007/10/)

-   ** [**  ](javascript:void(0)) [2006](http://ai.googleblog.com/2006/)
    -   [Feb](http://ai.googleblog.com/2006/02/)

    -   [Mar](http://ai.googleblog.com/2006/03/)

    -   [Apr](http://ai.googleblog.com/2006/04/)

    -   [Jun](http://ai.googleblog.com/2006/06/)

    -   [Jul](http://ai.googleblog.com/2006/07/)

    -   [Aug](http://ai.googleblog.com/2006/08/)

    -   [Sep](http://ai.googleblog.com/2006/09/)

    -   [Nov](http://ai.googleblog.com/2006/11/)

    -   [Dec](http://ai.googleblog.com/2006/12/)

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/icon18_wrench_allbkg.png)](https://www.blogger.com/rearrange?blogID=8474926331452026626&widgetType=BlogArchive&widgetId=BlogArchive1&action=editWidget&sectionId=sidebar "Edit")

[![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAihJREFUeNrsWa9Pw0AU7viRMDFRBAkzJDMIBIhJJhCzk7NILIqMv4AEhdz+BCY3OYssAlGBoAJREpZwAlHEBO8lr8nSvNeVbu1dyX3JlzTrXfa+u/e9d7c5joWFhYVO1Fa8PwH2gK6m+BRwAvSlAdsrgr8E1jUuMH73GTAEzrkBWymTewZlihhLmgDXIAFuHgGVQOUF7OSYM1p6PgTuA1vAZlUEvAnPdapcMY0VICECekQ0XRfYrqoHsAGNgXfAoMomRiFDEhOZkkL3S88hMaB2LwXp0bj+ps2edpToZpjfoIDQtBeU+xjoDzP2G/gCPKZ5f8WsCAFJoJgOCcFdWSTeL9YQMSvTA1h9BkI5jaiXhLpSCL/8mVZY0UpyJ9ZdOkniu1dmJ96BpzQu9w6s28gcOq9j6pwLdR8/36NK5CQKwJSMrb2MhhSglBpt4UjsrdsnNu0B3J0HCozbCc4TjyY2srEgos/4RQljCzNxl4ireQD8FOq+T+W0mTB2g7njhlR+Sy2jsXFvU658U8YTbeaGpdIu7mWkEAq5ZtIjIhFZdtfX7QHckSvB2B6zC3VdAkZk0kAQwaXTk/CzTXK3wjIExCs6ZJpTnE4uY1KV+KzFzA3KTiFPENHJkOPcsfpLhwe4btoSuvUqAR+6TOxlCE6ZfKUsJLgsqGW8OpqAGx2X+sLxrwUog+JUeQRMDBIwyXOcnlPtPnL0/UsT/8LnOxYWFhZG4leAAQAAQHEaYuzHbAAAAABJRU5ErkJggg==)](http://googleaiblog.blogspot.com/atom.xml)

Feed
----

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/icon18_wrench_allbkg.png)](https://www.blogger.com/rearrange?blogID=8474926331452026626&widgetType=HTML&widgetId=HTML6&action=editWidget&sectionId=sidebar "Edit")

Follow @googleai

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/icon18_wrench_allbkg.png)](https://www.blogger.com/rearrange?blogID=8474926331452026626&widgetType=HTML&widgetId=HTML5&action=editWidget&sectionId=sidebar-bottom "Edit")

Give us feedback in our [Product Forums](http://support.google.com/bin/static.py?hl=en&page=portal_groups.cs).

[![](./Google%20AI%20Blog_%20Moving%20Camera,%20Moving%20People_%20A%20Deep%20Learning%20Approach%20to%20Depth%20Prediction_files/icon18_wrench_allbkg.png)](https://www.blogger.com/rearrange?blogID=8474926331452026626&widgetType=HTML&widgetId=HTML1&action=editWidget&sectionId=sidebar-bottom "Edit")

[![](data:image/png;base64,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)](https://www.google.com/)

-   [Google](https://www.google.com/)
-   [Privacy](https://www.google.com/policies/privacy/)
-   [Terms](https://www.google.com/policies/terms/)

