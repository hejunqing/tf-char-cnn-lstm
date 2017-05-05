from __future__ import print_function
from __future__ import division


import tensorflow as tf

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in xrange(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]

    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
    input_ = tf.expand_dims(input_,-1)# hjq,(input_,1) origin

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1

            # [batch_size x max_word_length x embed_size x kernel_feature_size] origin
            conv = conv2d(input_, kernel_feature_size, kernel_size, embed_size, name="kernel_%d" % kernel_size)# hjq
            # conv=[batch_size x num_unroll_steps,reduced_length,1,kernel_feature_size] hjq
            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length,1, 1], [1, 1, 1, 1], 'VALID')#hjq,origin: [1,1,reduce_length,1]

            layers.append(tf.squeeze(pool))#origin:(pool,[1,2])

        if len(kernels) > 1:
            output = tf.concat(1, layers)
        else:
            output = layers[0]

    return output

class Model():
    def __init__(self, args, infer=False):

        self.kernels=[1,2,3,4,5,6,7]
        self.kernel_features=[50,100,150,200,200,200,200]
        assert len(self.kernels) == len(self.kernel_features), 'kernels size:%d,kernel_feature size:%d'%(len(self.kernels,len(self.kernel_features)))

        self.input_ = tf.placeholder(tf.int32, shape=[args.batch_size, args.num_unroll_steps, args.max_word_length], name="input")
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.num_unroll_steps], name='targets')
        target_list=tf.unpack(self.targets, axis=1)#hjq
        ''' First, embed characters '''
        with tf.variable_scope('Embedding'):
            char_embedding_r = tf.get_variable('char_embedding', [args.char_vocab_size,args.char_embed_size])
            char_embedinglist=tf.unpack(char_embedding_r)
            char_embedinglist[0]=tf.zeros([args.char_embed_size],dtype=tf.float32)
            self.char_embedding=tf.pack(char_embedinglist)
            # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
            input_embedded = tf.nn.embedding_lookup(self.char_embedding, self.input_)

            input_embedded_s = tf.reshape(input_embedded, [-1, args.max_word_length, args.char_embed_size])

        ''' Second, apply convolutions '''
        # [batch_size x num_unroll_steps, cnn_size]  # where cnn_size=sum(kernel_features)
        input_cnn = tdnn(input_embedded_s, self.kernels, self.kernel_features)

        ''' Maybe apply Highway '''
        if args.highway_layers > 0:
            input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=args.highway_layers)

        ''' Finally, do LSTM '''
        with tf.variable_scope('LSTM'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=True, forget_bias=0.0)
            if args.dropout > 0.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.-args.dropout)
            if args.rnn_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.rnn_layers, state_is_tuple=True)

            self.initial_rnn_state = cell.zero_state(args.batch_size, dtype=tf.float32)

            input_cnn = tf.reshape(input_cnn, [args.batch_size,args.num_unroll_steps , -1])
            # input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(1, num_unroll_steps, input_cnn)]
            input_cnn2=tf.unpack(input_cnn,axis=1)#hjq, a list of Tensor[batch_size x hidden],length of num_unroll_steps

            outputs, state = tf.nn.rnn(cell, input_cnn2,
                                                 initial_state=self.initial_rnn_state, dtype=tf.float32) #origin

            self.final_rnn_state=state

            # linear projection onto output (word) vocab
            self.logits = []
            with tf.variable_scope('WordEmbedding') as scope:
                for idx, output in enumerate(tf.unpack(outputs,axis=0)):
                    if idx > 0:
                        scope.reuse_variables()
                    self.logits.append(linear(output, args.word_vocab_size))

        self.loss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, target_list), name='loss')/args.batch_size
        cost=self.loss/args.num_unroll_steps
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')
        tvars = tf.trainable_variables()
        grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          args.max_grad_norm)
        optimizer=tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op=optimizer.apply_gradients(zip(grads,tvars))
