### 6. Probability & Statistics in Generative Models – Solve and analyze statistical problems related to generative models.

### Probability & Statistics in Generative Models

Generative models, such as **Generative Adversarial Networks (GANs)**, **Variational Autoencoders (VAEs)**, and **Normalizing Flows**, rely heavily on probability theory and statistics. Understanding statistical principles in these models helps improve their performance and interpretability.

### 1. **Key Concepts in Probability for Generative Models**

#### 1.1 **Probability Distribution**

Generative models aim to learn a **probability distribution** that describes the underlying data. For example:

- In **VAEs**, the model learns a distribution $p(x)$ over the data $x$.

- In **GANs**, the generator tries to produce data that follows the same distribution as the real data distribution.

- **Continuous Distribution**: Example: Gaussian distribution $p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$

- **Discrete Distribution**: Example: **Bernoulli Distribution** for binary outcomes.

#### 1.2 **Likelihood Function**

In many generative models, we compute the **likelihood** of data under a model. For example, in **Maximum Likelihood Estimation (MLE)**, we try to find the parameters $\theta$ that maximize the likelihood function $p(X | \theta)$.

- **MLE**: The likelihood function $L(\theta | X)$ for a dataset $X$ is:

  $$
  L(\theta | X) = \prod_{i=1}^{N} p(x_i | \theta)
  $$

  The parameters $\theta$ are estimated by maximizing $L(\theta | X)$.

#### 1.3 **Bayes' Theorem**

In **Bayesian Inference**, we use **Bayes' Theorem** to update the probability of a model’s parameters $\theta$ based on observed data $X$. This is often used in models like **Variational Autoencoders (VAEs)**:

$$
p(\theta | X) = \frac{p(X | \theta) p(\theta)}{p(X)}
$$

Where:

- $p(X | \theta)$ is the likelihood of the data.
- $p(\theta)$ is the prior belief about the parameters.
- $p(X)$ is the evidence, which is difficult to compute directly.

#### 1.4 **Entropy**

**Entropy** measures the uncertainty or randomness of a distribution. It is used in generative models like **VAEs** to quantify how well the model approximates the true distribution of data.

$$
H(p) = - \sum_{x} p(x) \log p(x)
$$

- **Lower entropy** indicates less uncertainty and more predictability.
- **Higher entropy** indicates more uncertainty and diversity.

---

### 2. **Statistical Problems in Generative Models**

#### 2.1 **Maximum Likelihood Estimation (MLE) in VAEs**

In **Variational Autoencoders**, the goal is to maximize the **Evidence Lower Bound (ELBO)**, which can be derived from the **log-likelihood** of the data. This involves a probabilistic approximation to the true posterior distribution of the latent variables.

The ELBO is given by:

$$
\text{ELBO}(\theta, \phi) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}[q(z|x) \| p(z)]
$$

Where:

- $q(z|x)$ is the approximate posterior distribution of latent variables.
- $p(x|z)$ is the likelihood of data given the latent variables.
- $p(z)$ is the prior distribution over latent variables (often a normal distribution).

The **KL Divergence** term measures the difference between the approximate posterior and the true prior.

#### 2.2 **KL Divergence vs. Wasserstein Distance in GANs**

In **GANs**, the goal is to minimize the difference between the distribution of generated data $p_{\text{gen}}$ and the real data distribution $p_{\text{real}}$. Two popular metrics for this task are:

1. **KL Divergence**: Measures how much information is lost when approximating one distribution by another.

   $$
   D_{KL}(p_{\text{real}} \| p_{\text{gen}}) = \int p_{\text{real}}(x) \log \frac{p_{\text{real}}(x)}{p_{\text{gen}}(x)} dx
   $$

2. **Wasserstein Distance (Earth Mover's Distance)**: Measures the minimum cost of transporting mass from one distribution to another.

   $$
   W(p_{\text{real}}, p_{\text{gen}}) = \inf_{\gamma \in \Gamma(p_{\text{real}}, p_{\text{gen}})} \mathbb{E}_{(x,y) \sim \gamma} \| x - y \|
   $$

Wasserstein Distance is often preferred in **Wasserstein GANs** due to its stability and ability to avoid vanishing gradients.

#### 2.3 **Evaluating Generative Models Using Statistical Tests**

When evaluating generative models, we use various statistical tests to assess how well the model generates data.

- **Kolmogorov-Smirnov Test (KS Test)**: Tests whether two samples come from the same distribution.
- **Chi-Square Test**: Compares the observed and expected frequencies of events.
- **Frechet Inception Distance (FID)**: Measures the distance between feature vectors of real and generated images. Lower FID indicates better quality of generated images.

---

### 3. **Statistical Tests for Generative Models**

Here is an example of how to use the **Kolmogorov-Smirnov (KS)** test to evaluate the distribution of generated data versus real data.

```python
import numpy as np
from scipy.stats import ks_2samp

# Example generated data (e.g., generated by GANs)
generated_data = np.random.normal(loc=0, scale=1, size=1000)

# Example real data
real_data = np.random.normal(loc=0, scale=1, size=1000)

# Perform Kolmogorov-Smirnov test
ks_statistic, p_value = ks_2samp(generated_data, real_data)

print(f"KS Statistic: {ks_statistic}")
print(f"P-Value: {p_value}")

# If p-value is small (e.g., < 0.05), we reject the null hypothesis (data distributions are different)
if p_value < 0.05:
    print("The generated data distribution differs from the real data distribution.")
else:
    print("The generated data distribution does not differ significantly from the real data distribution.")
```

### 4. **Analyzing Statistical Problems in GANs**

- **Bias-Variance Tradeoff**: Generative models face the issue of overfitting or underfitting the data. **High variance** leads to models that don’t generalize well, while **high bias** leads to poor training performance.
- **Mode Collapse**: In GANs, the generator may produce only a few types of data, leading to a collapse where the diversity of generated samples is low. This is a statistical issue where the generator doesn't learn the full distribution of the data.

---

### 5. **Practical Application: Using Generative Models for Statistical Inference**

Generative models can also be used for **statistical inference**:

- **Monte Carlo Methods**: Generative models like **VAEs** can be used to sample from complex distributions.
- **Sampling from Latent Space**: After training a **VAE** or **GAN**, we can generate new data by sampling from the latent space, which can be used for further statistical analysis.

For instance, generating synthetic data for rare events, estimating parameters from posterior distributions, or predicting the likelihood of future data points.

---

### Summary

- **Probability and Statistics** are integral to generative models as they involve estimating complex distributions and optimizing likelihoods.
- **KL Divergence**, **Wasserstein Distance**, and **BCE** are commonly used loss functions in models like **GANs** and **VAEs**.
- **Bayesian Inference** helps in incorporating prior beliefs into models, while **Entropy** quantifies uncertainty in generated data.
- **Statistical tests** like **KS Test** and **FID** help evaluate the quality of generative models.
