# Test 2nd derivative computation

import numpy as np
import tensorflow as tf

np.random.seed(0)

# loss = 'ctc'
loss = 'ce'

bs = 32
fdim = 26
ilen = 100
olen = 10
nlabels = 28
x = tf.Variable(np.random.rand(bs, ilen, fdim), dtype=tf.float32)
W = tf.Variable(np.random.rand(fdim, nlabels), dtype=tf.float32)
with tf.GradientTape() as g1:
    with tf.GradientTape() as g2:
        logits = tf.linalg.matmul(x, W)

        if loss == 'ctc':
            logits = tf.transpose(logits, [1, 0, 2])
            y = tf.Variable(np.random.randint(0, nlabels, (bs, olen)))
            loss = tf.reduce_mean(tf.nn.ctc_loss(y, logits, [olen] * bs, [ilen] * bs))
        elif loss == 'ce':
            y = tf.Variable(np.random.rand(bs, ilen, nlabels), dtype=tf.float32)
            loss = tf.nn.log_poisson_loss(y, logits)

        g2.watch(W)
        dl_dW = g2.gradient(loss, W)
    d = tf.linalg.norm(dl_dW)
    dd_dx = g1.gradient(d, x)
    print(dd_dx)