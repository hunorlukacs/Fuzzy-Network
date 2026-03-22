# Fuzzy Rule-Based Network (FRBN)

An interpretable and modular machine learning architecture designed to bridge the gap between high-performance nonlinear modeling and human-understandable decision-making.

## 📖 Overview
The **Fuzzy Rule-Based Network (FRBN)** functions as a "gray-box" model by integrating the hierarchical, layered principles of artificial neural networks (NN) with the transparency of fuzzy rule-based systems (FRBS). Unlike traditional "black-box" neural networks, the FRBN explicitly encodes knowledge through fuzzy if-then rules at each layer, providing direct insight into learned behaviors.



## ✨ Key Features
* **Interpretability**: Translates complex data into human-understandable fuzzy if-then statements.
* **Modularity**: Allows for reducing the number of parameters while maintaining accuracy within specific ranges.
* **Significance Scoring**: Includes a significance score to identify which specific rules are responsible for given data subdomains.
* **Domain Knowledge Integration**: Can be initialized using existing knowledge to accelerate training convergence.
* **Scalability**: Overcomes limitations in traditional systems by enabling deeper, layered fuzzy structures.

## 🏗 Architecture
The FRBN leverages a multi-layered structure where each layer systematically processes input features through a variable number of fuzzy rules.
* **Dimensionality**: Follows a customizable layered pattern (e.g., M-N-P-R).
* **Layer Structure**: Each layer consists of antecedents and consequents determined by its position in the sequence.

## 🚀 Optimization Algorithms
The model supports several optimization strategies to ensure stable convergence:
* **Hybrid Bacterial Memetic Algorithm (BMA)**: Combines global evolutionary search with local fine-tuning.
* **Adam**: Standard gradient-based optimization.
* **Levenberg-Marquardt and Trust-Region Levenberg-Marquardt**: This project utilizes the torch-levenberg-marquardt library [torch-levenberg-marquardt](https://github.com/fabiodimarco/torch-levenberg-marquardt) to implement efficient quasi-second-order optimization within the PyTorch framework. This project also implements an advanced trust-region-based version of LM.
  
## 📊 Performance & Validation
The FRBN has been validated across various environments, demonstrating accuracy and training times competitive with traditional neural networks:
* **Synthetic Benchmarks**: Tested on regression and multidimensional trigonometric datasets.
* **Industrial Application**: Validated on sensor datasets for manufacturing data analysis.

## 💻 Getting Started
The implementation is built using **PyTorch**.

### Prerequisites
* Python 3.x
* PyTorch
* torch-levenberg-marquardt (https://github.com/fabiodimarco/torch-levenberg-marquardt)

### Installation
```bash
git clone https://github.com/hunorlukacs/Fuzzy-Network
cd Fuzzy-Network
pip install -r requirements.txt
