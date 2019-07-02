# Automated Machine Learning with AutoKeras

Deep learning is a subset of machine learning focused on developing predictive models using neural networks, and allows humans to create solutions for object detection, image classification, speech recognition and more. One of the most popular deep learning libraries available is [Keras](https://keras.io), a high-level API to easily run TensorFlow, CNTK, and Theano. The main goal of Keras is to enable developers to quickly iterate and develop neural networks on multiple frameworks. 

Over the last year or so, there has been a lot of development in the AutoML space, which is why I have been writing so many blogs showing off different libraries. AutoML libraries have mostly focused around on traditional machine learning algorithms. Therefore, to take the Keras vision to the next level and increase the speed at which we can create neural networks, the Keras team as been developing the [AutoKeras](https://autokeras.com) library that aims to automatically learn the best architecture and hyper-parameters of a neural network to solve your specific need. 

Since the library is still in pre-release there are not a ton of resources to available when you start building a model with AutoKeras. Most of the examples show of the MNIST data set which is built into the Keras library. So while I do show a quick MNIST example in the demo, I also provide one with a custom image dataset that requires the developer to load images as numpy arrays prior to using them as input in the model training.

The demo I have created walks users through the process of:
- Curating your own image dataset
- Note that we will be using the [FastAI](https://www.fast.ai/) library, which is my favorite deep learning library and runs on top of PyTorch.
- You can also use the `data.zip` file available in the GitHub repository. 
- Train a model with Keras
- Train a model on the MNIST dataset using AutoKeras
- Train a model with downloaded images using AutoKeras


Overall, the autokeras library is rough. It does not work quite as Keras worked, which kind of threw me off, and a lot of the built in functions that makes Keras great are not available. I would not recommend using autokeras for any real neural network development, but the overall idea of using AutoML with Keras strings me greatly. Check out the demo I have provided on [GitHub](https://github.com/ryanchynoweth44/AutoMLExamples/blob/master/AutoKeras/demo/00_EnvironmentSetup.md). Please note that I developed the demo on a linux virtual machine, and that environment setup varies by environment. Additionally, GPU support will enable faster training times. 





