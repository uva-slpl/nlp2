# coding: utf-8
import numpy as np
import tensorflow as tf

def KL(kuma_a, kuma_b, beta_a, beta_b, terms=10):
    """
    Here I give you an example of how to code the KL between Kumaraswamy and Beta distributions.

    In the theory notebook: Kuma(alpha, beta), Beta(a, b)
    Here: Kuma(kuma_a, kuma_b), Beta(beta_a, beta_b).

    I am assuming at this point you have already clipped the parameters of 
        Kuma and Beta between 0.001 and 10 (for example), 
        to make sure we do not have 0s and to make sure we do not have large numbers.

    Note that:
        kuma_a, kumba_b, beta_a and beta_b should all have the same shape
    
    I am not doing it here, but I suggest you clip the resulting KL from below at 0 
        this prevents your optimiser from opportunistically exploiting numerical instabilities 
        due to the truncated Taylor expansion

    I hope this helps :)
    """
    kl = (kuma_a - beta_a) / kuma_a * (- np.euler_gamma - tf.digamma(kuma_b) - 1.0 / kuma_b)
    kl += tf.log(kuma_a * kuma_b) + tf.lbeta(tf.concat([tf.expand_dims(beta_a, -1), tf.expand_dims(beta_b, -1)], -1))
    kl += - (kuma_b - 1) / kuma_b
    # A useful identity: 
    #   B(a,b) = exp(log Gamma(a) + log Gamma(b) - log Gamma(a+b))
    # but here we simply exponentiate tf.lbeta instead, feel free to use whichever version you prefer
    betafn = lambda a, b: tf.exp(tf.lbeta(tf.concat([tf.expand_dims(a, -1), tf.expand_dims(b, -1)], -1)))
    # Truncated Taylor expansion around 1 
    taylor = tf.zeros(tf.shape(kuma_a))
    for m in range(1, terms + 1):  # m should start from 1 (otherwise betafn will be inf)!
        taylor += betafn(m / kuma_a, kuma_b) / (m + kuma_a * kuma_b)
    kl += (beta_b - 1) * kuma_b * taylor
    return kl
