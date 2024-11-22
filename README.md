# Differential Privacy for Deep Learning with Flask Deployment
## Overview
This repository demonstrates the implementation of a deep learning model trained with Differential Privacy (DP) and deployed using a Flask API. The project utilizes TensorFlow Privacy (TFP) to ensure data privacy while maintaining model utility.This repository demonstrates the implementation of a deep learning model trained with Differential Privacy (DP) and deployed using a Flask API. The project utilizes TensorFlow Privacy (TFP) to ensure data privacy while maintaining model utility.
## Features

* Implementation of differentially private stochastic gradient descent (DPSGD) for model training
* Deployment of the trained model via a Flask API for inference
* Resources and links to further readings, tutorials, and relevant code libraries
## Repository Structure
<pre>
DifferentialPrivacy
├── Dp.py
└── app.py
└── requirements.txt
└── README.md
</pre>

## How to Use

### Setup

1. Clone the repository and navigate to its directory:
<pre>
git clone https://github.com/BVChandrahaas/DifferentialPrivacy.git
cd DifferentialPrivacy
</pre>
2. Install the required dependencies:
<pre>
pip install -r requirements.txt
</pre>
3. Train the Model
Run the dp_training.py script to train the model with differential privacy:
<pre>python Dp.py</pre>
The trained model will be saved in the saved_model/ directory.
Run the Flask App
Start the Flask API server:
<pre>python app.py</pre>
By default, the app runs at http://127.0.0.1:5000.

## Resources

* **Papers**
	+ Deep Learning with Differential Privacy by Abadi et al.
	+ Federated Learning with Differential Privacy by McMahan et al.
* **Code Libraries**
	+ TensorFlow Privacy
	+ PyTorch-DP
* **Tutorials**
	+ Differential Privacy Tutorial by Stanford Natural Language Processing Group
	+ Federated Learning with Differential Privacy by TensorFlow Team
## Contributing

We welcome contributions! Feel free to:

* Fork this repository
* Submit pull requests
* Report issues

Your contributions will help improve Vanilla Split Learning.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
