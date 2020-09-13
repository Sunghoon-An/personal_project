# Increasing Interpretability of XAI using Generalized Least Square Estimator 
Layer-wise Relevance Propagation (LRP) is widely used as an AI postinterpreting method. However, the LRP derives the relevance score of neurons in each layer of the network and the relevance score of the input variable, without considering their correlation with the heteroskedasticity of the estimated weights. 
In this study, to increase the reliability of the relevance score, the dependency score of the neurons in the previous layer was determined as the dependent variable and the relevance scores of the neurons in the next layer associated with that neuron were determined as explanatory variables to re-estimate the dependent variable using GLS.
The process was carried out sequentially from the output layer to the input layer to ultimately increase the reliability of the relevance score of the input variable.
As a result of applying the proposed methodology to MNIST data, the estimated relevance score of the visualized input variable was more similar to the original test image than LRP Heatmap, and the structural similarity index improved by up to 22% over the index of the LRP relevance score. 
It was also confirmed that the analytical power derived from RemOve And Retrain(ROAR) and Keep And Retrain(KAR) was enhanced over LRP.
## [Article]http://dcollection.yonsei.ac.kr/public_resource/pdf/000000525518_20200913132241.pdf

[TODO-list] 
- [x] abstract
- [ ] Guide-line
- [ ] install 방법
- [ ] Run
- [ ] IP
