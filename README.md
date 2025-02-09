# Generative Adversarial Networks (GANs) Implementation

## Dillon McCarthy

## Overview
This project implements and experiments with Generative Adversarial Networks (GANs), including modifications to the generator, image saving during training, and adapting the model for the CIFAR-10 dataset.

## Features
- **Enhanced Generator:** Additional convolutional layers in the generator for improved image synthesis.
- **Image Saving:** Saves generated images every 10 epochs to track progress.
- **Dataset Update:** Replaces MNIST with CIFAR-10, modifying the network to handle color images.

## Implementation Details

### Data Preparation
- Load and preprocess the CIFAR-10 dataset (32x32 color images).
- Normalize images to the range [-1, 1] for stable training.

### Model Architecture
#### Generator
- Converts random noise (`latent_size = 100`) into 32x32 pixel images.
- Uses transposed convolution layers (`Conv2DTranspose`) for upsampling.
- Includes batch normalization and LeakyReLU activation.

#### Discriminator
- Classifies images as real or fake.
- Uses convolutional layers (`Conv2D`) to extract features.
- Ends with a dense layer outputting a probability score.

### Loss Functions
- **Discriminator Loss:** Binary cross-entropy comparing real and fake classifications.
- **Generator Loss:** Binary cross-entropy pushing generated images to be classified as real.

### Training Process
- The discriminator is trained on real and generated images.
- The generator is updated based on how well it fools the discriminator.
- Images are saved every 10 epochs to visualize improvements.

## Summary

The models were trained with limited compute resources, but after free allocated time on a Google Colab GPU, the generator is able to generate colorful, low quality images.  The losses of the generator and discriminator appear close to an equillibrium and both show signs of future improvement.  In order to improve the generator and the training process, the models could be trained for longer but with a fine-tuned dropout hyperparameter among others to mitigate mode collapse.

## Questions
**Q1: Explain the minimax loss function in GANs and how it ensures competitive training between the generator and discriminator.**

The minimax loss function creates a game where the generator and discriminator compete against each other. The generator attempts to minimize the loss, while the discriminator attempts to maximize it. In this way, the two models compete against each other and learn simultaneously. Both models use the fake generated data to calculate the gradients and perform backpropagation, so as the discriminator learns to better distinguish between real and fake images, it forces the generator to learn to generate better images.

**Q2: What is mode collapse, Why can mode collapse occur during GAN training? and how can it be mitigated?**

Mode collapse occurs when the generator converges on one type of image and generates similar images. In this way, it fails to capture the diversity of the true dataset and does not generate a variety of images representative of the true distribution. Several approaches can be used to mitigate mode collapse, such as dropout, adding Gaussian noise to input data, or using variant architecture such as Wasserstein GAN (WGAN) which uses Wasserstein distance and may introduce gradient penalties.

**Q3: Explain the role of the discriminator in adversarial training?**

The discriminator serves as the classifier in this problem, seeking to differentiate real from fake data. It is important that it is trained at a similar pace to the generator, however, so that the generator can still learn something. The discriminator therefore needs to learn how to differentiate the images over time, initially just guessing which is which and quickly learning the patterns and shapes of the true distribution so that the generator can seek to close that gap and more closely model that true distribution.

**Q4: How do metrics like IS and FID evaluate GAN performance?**

Inception Score (IS) is used to automatically score the quality of a simulated image using a pre-trained Inception v3 Network classifier that outputs the probability of a class. Using the confidence and KL divergence, the metric is able to accurately rate the quality of a generated image similar to human evaluations.

Frechet Inception Distance (FID) also uses a pre-trained Inception v3 model to calculate the distance between the real and fake feature vectors using their mean and covariance.

Both of these metrics are useful for evaluating GAN performance because they closely replicate and automate human scoring of images. Metrics such as accuracy obviously do not work with content generation, so oftentimes human evaluation is required. These metrics automate this human evaluation using pre-trained models to accurate assess the quality of the generated images from the GAN for comparing different architectures and performances.

