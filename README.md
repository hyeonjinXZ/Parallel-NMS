# Parallel-NMS
##### Hyeonjin Lee, Jeong-Sik Lee, and Hyun-Chul Choi
##### IEEE ACCESS https://ieeexplore.ieee.org/document/9646917
##### [non_maximum_supression.py](https://github.com/hyeonjinXZ/Parallel-NMS/blob/main/non_maximum_supression.py)

Non-maximum suppression (NMS) is an unavoidable post-processing step in the object detection pipeline. We propose a parallel computation method using GPU multi-cores to compute faster than the previous NMS. We drastically reduced the complexity from $O(N^2)$ to $O(N)$ and the time consumption of NMS to be applied to real-time detection with negligible degradation of detection performance and very slight additional memory consumption. Furthermore, when there is a small number of overlapped objects, our parallel NMS achieved an improvement in precision.

![alt text](https://github.com/hyeonjinXZ/Parallel-NMS/blob/main/parallel_nms.png "Parallel_NMS")

# Citation
If you find this code useful, please consider citing:

@ARTICLE{9646917,
  author={Lee, Hyeonjin and Lee, Jeong-Sik and Choi, Hyun-Chul},
  journal={IEEE Access}, 
  title={Parallelization of Non-Maximum Suppression}, 
  year={2021},
  volume={9},
  number={},
  pages={166579-166587},
  doi={10.1109/ACCESS.2021.3134639}}
