\# 3D Reconstruction Dataset and Evaluation



This repository contains datasets, reconstructed 3D models, evaluation results, and scripts used for the reconstruction and analysis of different objects.



The repository was created as part of a master's thesis project.



\---



\# Objects



The following objects are included in the dataset:



\- Cuboid

\- NutM20

\- Benchy



\---



\# Image Data



The repository contains:



\- RGB images captured using a OnePlus smartphone camera

\- RGB and depth images captured using an Intel RealSense camera



\---



\# Reconstruction Methods



The repository includes reconstructed meshes and results generated using multiple reconstruction approaches and external frameworks, including:



\- Visual Hull

\- COLMAP

\- Meshroom

\- Point2CAD

\- TRELIS

\- CAD-Recode



Additionally, mixed and uncleaned reconstructed models are included for comparison and evaluation purposes.



\---



\# Evaluation Results



The evaluation results include:



\- Chamfer Distance

\- Hausdorff Distance

\- Runtime Analysis



\---



\# Scripts



\## visual\_hull.py



Python implementation of the Visual Hull reconstruction pipeline.



\## collage.py



Utility script for arranging and visualizing image results.



\---



\# Requirements



Recommended environment:



\- Python 3.8+

\- Open3D

\- NumPy

\- OpenCV

\- Matplotlib



\## Usage



The usage of this script is documented within the README in the Scripts Folder.



\# License



This repository is intended for academic and research purposes.

