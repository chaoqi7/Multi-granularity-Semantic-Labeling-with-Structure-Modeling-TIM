# Multi-granularity-Semantic-Labeling-with-Structure-Modeling-TIM
Multi-granularity Semantic Labeling of Point Clouds for the Measurement of the Rail Tanker Component with Structure Modeling（IEEE TIM）
# Platform
python3.6
pytorch0.3
# Code structure
* `./Structure Modeling/*` - used for structure modeling for tanker point cloud.
* `./Semantic Labeling/*` - used for semantic labeling  for tanker point cloud based on structure modeling.
## Processes 
This project should use files in "Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs(spg)" and “Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs”.  
  1.Use Cut Pursuit to cmake the files in structure modeling.  
  2.Deploy SPG and copy the files in semantic labeling to "learning" in spg.  
## Reference By
[loicland/cut-pursuit](https://github.com/loicland/cut-pursuit)<br>
[loicland/superpoint_graph](https://https://github.com/loicland/superpoint_graph)<br>
## Citation
Multi-granularity Semantic Labeling of Point Clouds for the Measurement of the Rail Tanker Component with Structure Modeling(IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT 2020)
