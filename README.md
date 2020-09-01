# Depth Prediction [(Google AI)](https://ai.google/)
# Moving Camera, Moving People: A Deep Learning Approach to Depth Prediction
The human visual system has a remarkable ability to make sense of our 3D world from its 2D projection. Even in complex environments with multiple moving objects, people are able to maintain a feasible interpretation of the objects’ geometry and depth ordering. The field of computer vision has long studied how to achieve similar capabilities by computationally reconstructing a scene’s geometry from 2D image data, but robust reconstruction remains difficult in many cases.

A particularly challenging case occurs when both the camera and the objects in the scene are freely moving. This confuses traditional 3D reconstruction algorithms that are based on triangulation, which assumes that the same object can be observed from at least two different viewpoints, at the same time. Satisfying this assumption requires either a multi-camera array (like Google’s Jump), or a scene that remains stationary as the single camera moves through it. As a result, most existing methods either filter out moving objects (assigning them “zero” depth values), or ignore them (resulting in incorrect depth values).





```
Thursday, May 23, 2019
Posted by Tali Dekel, Research Scientist and Forrester Cole, Software Engineer, Machine Perception
