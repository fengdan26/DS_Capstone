Graph-Based Stock Market Analysis with GANs
This repository contains the implementation of our project on Graph-Based Stock Market Analysis using Generative Adversarial Networks (GANs). The goal of this project is to predict future stock market trends by generating realistic candlestick data through innovative GAN architectures and techniques.

Project Overview
The project explores various GAN architectures and enhancements to improve stock market predictions:

RGB Stick Representation: Encodes stock data into a visual format, integrating price trends and trading volume for richer inputs and outputs.
CGAN Variants:
CGAN: Baseline implementation.
CGAN + LSTM: Incorporates LSTM to capture temporal dependencies in stock trends.
CGAN with Numbers: Focuses on numerical representation of stock data.
Training is performed on high-performance computing (HPC) environments and local machines.
Repository Structure
cgan.py: Implementation of the baseline CGAN model.
cgan+lstm.py: Implementation of the CGAN model with LSTM for enhanced temporal learning.
cgan_num6.py: Implementation of the GAN variant for numerical data analysis.
making_RGB_Stick5.py: Script for generating RGB Stick representations of stock data.
utils.py: Utility functions for data preprocessing and model evaluation.
__pycache__/: Cached files for faster execution.
Setup Instructions
Clone the Repository:

git clone https://github.com/fengdan26/Graph_Based_Stock_Market_Analysis_with_GANs.git
cd Graph_Based_Stock_Market_Analysis_with_GANs
Install Dependencies: Install the required Python libraries using pip:

pip install -r requirements.txt
Prepare Data: Ensure your stock market data is preprocessed and formatted to match the input requirements for RGB stick generation.

Run Models:

To run the baseline CGAN model:
python cgan.py
To run the CGAN + LSTM model:
python cgan+lstm.py
To run the GAN with numbers:
python cgan_num6.py
Key Features
Data Representation: Innovative RGB stick encoding of multi-day stock trends and trading volume.
Multiple Architectures: Comprehensive evaluation of CGAN, CGAN + LSTM, and GAN with numbers.
Scalable: Supports both HPC environments and local execution.
Results
Our models demonstrate strong alignment with target stock trends, capturing both temporal dependencies and structural patterns in stock data. For detailed results and visual assessments, refer to the project's documentation.
Authors
Peiquan Feng: Local training and CGAN + LSTM development.
Kaiwen Hu: HPC training and GAN with Numbers development.

Acknowledgements
We acknowledge the contributions of various open-source libraries and frameworks that made this project possible.
