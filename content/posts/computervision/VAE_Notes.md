---
title: "Variational AutoEncoder Concepts"
math: true
---

<!--more-->

# Brief Intro - AutoEncoders

As the name suggests, Variational AutoEncoders (VAEs) belong to the broader family of AutoEncoders. There are several types of AutoEncoders, each designed to address different problem statements. In this section, our focus will be on Variational AutoEncoders. But before that, let’s briefly touch upon the general concept of AutoEncoders and the limitations that VAEs aim to address.

Common types of AutoEncoders include:

 - AutoEncoders
 - Deniosing AutoEncoders
 - Sparse AutoEncoders
 - Variational AutoEncoders

## AutoEncoder Process

All AutoEncoder models consist of three main components: Encoder, Latent Space, and Decoder.
The Encoder compresses the input data(X) into a compact representation (latent space-Z).
This latent representation(Z) is then passed to the Decoder, which attempts to reconstruct the original input(X).

The process can be summarized mathematically as:

Z = Encoder(X)  
X' = Decoder(Z)  
Loss = Σ(X - X')²

## Issues with AutoEncoders

During training, the encoder compresses input data into a latent space. However, this latent space is often non-continuous and non-regularized. If we sample random points from this space and pass them through the decoder, when we pass the random points from latent space decoder might not be able to generate a meningful image, since the decoder hasn't learned how to handle unseen data

As a result:

- Standard AutoEncoders are not well-suited for generative tasks (i.e., creating new data).

- They are primarily useful for dimensionality reduction and input compression.




# Terminology Required for VAE

Variational AutoEncoders (VAEs) are designed to address key limitations of traditional AutoEncoders. As discussed earlier, the latent space learned by standard AutoEncoders is not continuous or well-structured, making them ineffective for generating new data.

VAEs overcome this by regularizing the latent space to follow a continuous and structured distribution — typically a Gaussian distribution  𝑁(0,𝐼).To understand how this is achieved, it is important to understand few fundamental concepts from Bayesian probability theory.

Let’s explore these concepts using an intuitive example and relate them to the context of VAEs.

Let’s explore these concepts using an intuitive example and relate them to the context of VAEs.

## Bayesian Probability Concepts:

Let’s use an analogy:

X = Observed Data(e.g., a prepared dish)
Z = Latent Variable (e.g., a recipe)

**P(Z|X) = Posterior Probability**

Given the dish (X), what is the probability that it was prepared using recipe
Z


**P(Z) = Prior Probability**

What is the probability that recipe Z exists in the chef's recipe book(irrespective of any Dish)


**P(X|Z) = Likelihood**

Given a specific recipe Z, what is the probability of producing the observed dish X


**P(X) = Marginal Probability**
What is the total probability of observing the dish X, across all possible recipes

Since many different recipes could potentially produce a similar dish, we compute this by averaging (integrating or summing) over all recipes: This is also called marginal likelihood and acts as a normalizing constant in Bayes’ Theorem.

P(X) = Σ P(X|Z) * P(Z)

Baye's Theorem  P(Z|X) =    \$P(X|Z) * P(Z)/ P(X)\$
                            

                


### Bayesian Probabilty for VAE:

**P(Z|X) = Posterior Probability**

Given the image \( X \), this is the probability that it was generated from the latent distribution \( Z \).  

→ In the context of a VAE, this is modeled by the **encoder**: it approximates ( P(Z|X) \) using \( q(Z|X) \).

** P(Z) = Prior Probability **

This is our assumption about the distribution of latent variables \( Z \) before seeing any data.  
→ In VAEs, this is typically assumed to be a **standard Gaussian** 𝒩(0, I), defining a smooth latent space.

**P(X|Z) = Likelihood**

Given a latent variable \( Z \), this is the likelihood of generating the image \( X \).  

→ In a VAE, this is modeled by the **decoder**: it tries to reconstruct \( X \) from \( Z \).

**P(X) = Marginal Probabitlity**

This is the total probability of observing image \( X \), integrating over all possible latent variables \( Z \).  

→ The **goal of the VAE** is to maximize this value (maximize likelihood of the data).


P(X) = ∫ P(X|Z) * P(Z) * d(Z)

This integral is computationally intractable (i.e., not feasible to evaluate directly due to the infinite possibilities over the continuous latent space Z). Therefore, VAEs optimize the **Evidence Lower Bound (ELBO)** on log P(X), which we will cover in the next section.

**Example:**

Suppose the latent space Z is a 1-dimensional real-valued variable.

Let's assume:

Z ~ 𝒩(0,1) Standard Normal Distribution
X is an image (say, a handwritten digit), and the model learns P(X|Z)

Now to compute P(X) = ∫ P(X|Z) * P(Z) * d(Z)

You would need to consider every possible value of Z to calculate this integral. But since: The real number line is continuous and uncountably infinite, there are infinitely many values Z ∈ (−∞,∞)

we can't sum or evaluate each one explicitly — hence, it's intractable.

Even in 1D, there are infinite values of Z to integrate over.

In practice, Z is often multi-dimensional, e.g.,
$R^{20}$, $R^{100}$ etc.

Hence the integral becomes a high-dimensional integral over an infinite space — something that is impossible to compute exactly without approximation



        







# Evidence LowerBound

The objective of a VAE is to generate images from a latent space that follows a continuous Gaussian distribution. This allows the model to produce meaningful samples—either from the training distribution or newly generated ones—by learning a smooth latent space, in other words  Maximizing the Probability of observing X under generative Model


Marginal Likelihood :- $P(X), log P(X)$

In the context of VAEs, we deal with two distributions:

 - $P(X)$: Marginal distribution of observed data
 - $P(Z)$: Prior distribution over latent variables

While the data distribution P(X) may not be Gaussian, we impose a Gaussian prior on the latent space P(Z) through training, we aim to learn a mapping such that the encoder transforms input data into a latent space distribution that approximates this prior.



<img src="vae.jpg" alt="VAE Distributions" width="600"/>

## Deriving the Evidence Lower Bound (ELBO)

We know that whole process comprise of 3 components Encoder, Latent Space & Decoder and from Bayesian Proabitility we know that to calculate

$$P(X) = ∫ P(X|Z) * P(Z) dZ$$

$$P(Z \mid X) = \frac{P(X|Z) * P(Z)}{P(X)}$$

According to Bayesian principles, the marginal likelihood P(X) is intractable due to the integral over infinitely many possible latent variables
Similarly, the posterior $P(Z|X)$ which depends on $P(X)$ is also intractable. However, in order to compute the loss and perform backpropagation in a Variational Autoencoder, we require access to $P(Z|X)$.To overcome this limitation, we approximate the true posterior $P(Z|X)$ with a tractable variational distribution $Q(Z|X)$ which is learned by the encoder.
      


<div style="text-align: left;">
$$
P(Z|X) \approx Q(Z|X)
$$
</div>

<div style="text-align: left;">
$$
P(X) = \int P(X|Z) P(Z) dZ
$$
</div>

<div style="text-align: left;">
$$
P(Z|X) = \frac{P(X|Z) P(Z)}{P(X)}
$$
</div>

<div style="text-align: left;">
$$
P(X) = \int P(X|Z) P(Z) dZ
$$
</div>

<div style="text-align: left;">
$$
\log P(X) = \log \int P(X|Z) P(Z) dZ \quad \text{(Its common to take log for numerical stability)}
$$
</div>

<div style="text-align: left;">
$$P(X, Z) = P(X|Z) P(Z) \quad \text{(From Chain Rule of Probability)}$$
</div>


Hence $log P(X)$ can be re-written as below

<div style="text-align: center;">
$$
\begin{aligned}
\log P(X) &= \log \int P(X,Z)dZ \\
\log P(X) &= \log \int \frac{Q(Z|X)}{Q(Z|X)} P(X,Z) dZ \\
\log P(X) &= \log \int \frac{P(X,Z)} {Q(Z|X)} * Q(Z|X) * dZ
\end{aligned}
$$
</div>


$$\mathbb{E}_{q(z|x)} f(Z)  = ∫ f(Z) * Q(Z|X) * dZ \quad \textbf{(From Expection Formula)}$$

$$f(Z) = \frac{P(X,Z)}{Q(Z|X)}\$$ 

thus effectively we can write logP(X) as below 

<div style="text-align: center;">
$$
\log P(X) = \log \mathbb{E}_{q(z|x)}\left[\frac{P(X,Z)}{q(z|x)}\right]
$$
</div>



<div style="text-align: center;">
$$
\log \mathbb{E}_{Q(Z|X)}\left[\frac{P(X,Z)}{Q(Z|X)}\right] \geq \mathbb{E}_{Q(Z|X)}\left[\log \frac{P(X,Z)}{Q(Z|X)}\right] \quad \text{(Jenson's Inequality)}
$$
</div>

<div style="text-align: center;">
$$
\textbf{Equation of ELBOW:- } \log P(X) \geq \mathbb{E}_{q(z|x)}\left[\log \frac{P(X,Z)}{Q(Z|X)}\right]
$$
</div>

<div style="text-align: center;">
$$
\textbf{From Bayes Theorem:-} P(X,Z) = P(X|Z) \cdot P(Z)
$$
</div>

<div style="text-align: center;">
$$
\textbf{Expanding the Expectation:- } \mathbb{E}_{q(z|x)}\left[\log \frac{P(X,Z)}{Q(Z|X)}\right] = \mathbb{E}_{q(z|x)}\left[\log \frac{P(X|Z) \cdot p(Z)}{q(z|x)}\right]
$$
</div>


<div style="text-align: center;">
$$
\begin{aligned}
    \mathbb{E}_{Q(Z|X)}\left[\log \frac{P(X|Z) \cdot P(Z)}{Q(Z|X)}\right] &= \mathbb{E}_{Q(Z|X)}[\log P(X|Z)] \\
    &\quad + \mathbb{E}_{Q(Z|X)}\left[\log \frac{P(Z)}{Q(Z|X)}\right]
\end{aligned}
$$
</div>



<div style="text-align: center;">
$$ \textbf{From KL Divergence:-}
\text{KL}(q(z|x)||p(z)) = \mathbb{E}_{q(z|x)}\left[\log \frac{q(z|x)}{p(z)}\right] = -\mathbb{E}_{q(z|x)}\left[\log \frac{p(z)}{q(z|x)}\right]
$$
</div>



<div style="text-align: center;">
$$\textbf{Final ELBO Expression:-}
\log P(X) \geq \underbrace{\mathbb{E}_{Q(Z|X)}[\log P(X|Z)]}_{\text{Reconstruction Loss}} - \underbrace{\text{KL}(Q(Z|X)||P(Z))}_{\text{KL Divergence}}
$$
</div>



**Intution behind ELBOW:-** The marginal likelihood $logP(X)$ is intractable due to above discussed reasons, the integral over all possible latent variables
Z is computationally expensive and often not solvable analytically.

However, our goal is to maximize P(X) which corresponds to training the model to better generate observed data.Generating Observed data with the trained Model(parameters) , Since we cannot maximize $log P(X)$ directly, we instead construct a tractable surrogate function that serves as a lower bound to it — the Evidence Lower Bound (ELBO).

If we are able to maximize this lower bound, we are guaranteed to maximize the value of $ log P(X)$


**Analogy**:- Imagine trying to measure the depth of the deepest part of an ocean, but due to dangerous conditions, it's impossible to access directly.

Instead, you find a nearby ocean that is similar in structure and composition, but shallower and safe to measure.

Once you determine the depth of this nearby ocean, you at least know that the original ocean is deeper than this — giving you a lower bound on its true depth.

In this analogy:

- The original ocean = $log P(X) (true but intractable log-likelihood)
- The reference ocean = ELBO (tractable, approximated lower bound)
- Maximizing the reference depth = optimizing the ELBO during training

Also, This lower bound becomes tighter as our variational approximation $Q(Z|X)$
becomes closer to the true posterior $P(Z|X)$ which is measured via KL divergence. When the KL divergence goes to zero, ELBO equals $log P(X)$


```python

```



# Reparameterization Trick

## Necessity for Reparameterization:

If we revisit the VAE training process, it consists of three key steps:

- The input is passed through an encoder that outputs the parameters of a latent distribution.

- A sample is drawn from this latent space

- The sampled latent variable is passed to the decoder to reconstruct the output.

Like any machine learning model, a VAE learns its objective through training. In early iterations, it performs poorly and improves gradually by minimizing a loss function that guides the parameter updates (weights and biases).

In the context of VAEs, there are two objectives, hence two associated loss terms:

**Reconstruction Loss**

The decoder should be able to reconstruct the original input (e.g., image) or generate a meaningful approximation of it. During training, the decoder’s output is compared with the input passed to the encoder. This forms the reconstruction loss, which is typically measured using Cross-Entropy Loss, Mean Squared Error (MSE), or Binary Cross-Entropy (BCE).

**KL Divergence Loss**

The latent space learned by the encoder should follow a standard Gaussian prior distribution. To enforce this, we measure how much the encoder's distribution $Q(Z|X)$ deviates from prior $P(Z)$ which is usually 𝓝(0,I). This difference is calculated using the KL divergence.

**Challenge: Sampling Breaks Backpropagation**

While calculating the KL divergence and reconstruction loss, we sample from the latent space to generate inputs for the decoder. However, sampling is a non-differentiable operation, which means gradients cannot flow backward through it during training.

This breaks the end-to-end learning pipeline — the encoder cannot learn how to adjust its parameters to reduce the overall loss.

**Solution:Reparameterization Trick**
To solve this, we reparameterize the random variable z as a deterministic function of a random noise and learnable parameters:

$$ Z = μ(X) + σ(X) * ϵ  ,\quad ϵ ≈ 𝒩(0,I) $$

This separates the randomness $ϵ$ from the network parameters (μ,σ). Now, the sampling step becomes differentiable with respect to μ and σ, allowing gradients to flow through and the encoder to be trained via backpropagation.


```python

```

# Final Loss Function

We start with our Loss function that has two components , Reconstruction Loss & KL Divergence Loss


<div style="text-align: center;">
$$
\textbf{ELBO Expression:-}\quad \log P(X) \geq \underbrace{\mathbb{E}_{Q(Z|X)}[\log P(X | Z)]}_{\text{Reconstruction Loss}} - \underbrace{\text{KL}(Q(Z|X)||P(Z))}_{\text{KL Divergence}}
$$
</div>

We have closed form for KL Divergence between, to calculate difference between two distributions

$Q(Z|X) = 𝒩(μ,σ^2) - \textbf{encoder output (approximate posterior)}$ \

$P(Z) = 𝒩(0,I) - \textbf{Prior}$

$$
D_{\text{KL}} \left( 𝒩(\mu, \sigma^2) \,\|\, 𝒩(0, 1) \right)
= \frac{1}{2} \sum_{i=1}^{d} \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)
$$

The KL divergence measures how far $Q(Z|X)$ is from Prior $P(Z)$ per dimension of the latent space. Since the prior $P(Z)$ is fixed to standard normal 𝒩(0,1) and the encoder gives us μ(X) & σ(X) these are captured from encoders' output

The final loss function is

<div style="text-align: center;">
$$
- \underbrace{\mathbb{E}_{Q(Z|X)}[\log P(X | Z)]}_{\text{Reconstruction Loss}} + \underbrace{\left(\frac{1}{2} \sum_{i=1}^{d} (\mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1)\right)}_{\text{KL Divergence}}
$$
</div>


The original ELBO is a lower bound we try to maximize.But most optimizers minimize loss, so we negate the ELBO to convert it into a minimization objective.



# Few Important Points on VAE's choices of using Gaussian Distribution

## Why compare the latent space with a Gaussian

In VAEs, we force the latent space Z to follow a known, simple distribution, typically the standard normal distribution $𝒩(0,I)$ even though the true posterior over $P(Z|X)$ might be complex.

The latent space is intended to be a compressed representation of your data distribution. For example:

 - A cat image X gets encoded into some point Z in latent space.
 - Ideally, similar inputs (like other cat images) map to nearby regions in that space.

## Why force latent space to resemble

A. If we let the encoder produce arbitrary Q(Z|X) the latent space would become non-continuous or clustered in strange regions.

By constraining all encodings to be close to a standard Gaussian, we ensure the entire latent space is:

 - Dense (no holes)
 - Continuous (no abrupt jumps)
 - Interpolatable (you can walk between two points and get meaningful outputs)

B. Regularization

If we only trained the autoencoder to minimize reconstruction loss, it might overfit: memorize  X-> Z-> X but produce garbage when sampling new Z.

C.Generative Capability

Once the model is trained, we can sample $ Z ≈ 𝒩(0,I) $ and feed to decoder to generate new, meaningful data.

Without the Gaussian prior, this would be impossible — we wouldn’t know where to sample from.

## But why Gaussian and not other distributions

- Gaussian distributions are mathematically convenient — they have a closed-form KL divergence.

- The reparameterization trick works smoothly with Gaussians.

- Sampling and interpolating from Gaussians is straightforward.

However in advanced VAE's You can use more complex priors (like Gaussian Mixture Models, VampPriors, flows, etc)

These are used when the data structure is more complicated than what a unimodal Gaussian can capture.

Analogy:

Imagine you're organizing books into shelves:

- Without constraints: Books are scattered randomly — you don’t know where to find a particular topic.

- You force all books to be sorted neatly by topic on a continuous, ordered shelf.

Now we can pick a point (sample Z) and be confident it will lead you to a coherent topic (decode to a valid image or text).
