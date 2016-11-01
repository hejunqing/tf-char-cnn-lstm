from __future__ import absolute_import #import local module needs the absolute path
from __future__ import division #make the int division produce float result
from __future__ import print_function #make the print a function, print('str')

import os
import time
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, DataReader


flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_float  ('learning_rate',       1.0,  'starting learning rate')
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          20,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          25,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_integer('print_every',    100,    'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
flags.DEFINE_integer('gpuid',   0, 'which gpu to use,-1 means use cpu')
FLAGS = flags.FLAGS


def run_test(session, m, data, batch_size, num_steps):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)
  
    for step, (x, y) in enumerate(reader.dataset_iterator(data, batch_size, num_steps)):
        cost, state = session.run([m.cost, m.final_state], {
            m.input_data: x,
            m.targets: y,
            m.initial_state: state
        })
        
        costs += cost
        iters += 1

    return costs / iters


def main(_):


    ''' Trains model from data '''

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)

    train_reader = DataReader(word_tensors['train'], char_tensors['train'],
                              FLAGS.batch_size, FLAGS.num_unroll_steps)

    valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'],
                              FLAGS.batch_size, FLAGS.num_unroll_steps)

    test_reader = DataReader(word_tensors['test'], char_tensors['test'],
                             FLAGS.batch_size, FLAGS.num_unroll_steps)

    print('initialized all dataset readers')

    if FLAGS.gpuid>=0:
        device_name="/gpu:"+str(FLAGS.gpuid)
    else:
        device_name="/cpu:"+str(FLAGS.gpuid)

    g=tf.Graph()#hjq
    with tf.device(device_name),g.as_default():#hjq

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = model.inference_graph(
                char_vocab_size=char_vocab.size,
                word_vocab_size=word_vocab.size,
                char_embed_size=FLAGS.char_embed_size,
                batch_size=FLAGS.batch_size,
                num_highway_layers=FLAGS.highway_layers,
                num_rnn_layers=FLAGS.rnn_layers,
                rnn_size=FLAGS.rnn_size,
                max_word_length=max_word_length,
                kernels=eval(FLAGS.kernels),
                kernel_features=eval(FLAGS.kernel_features),
                num_unroll_steps=FLAGS.num_unroll_steps,
                dropout=FLAGS.dropout)
            train_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps, FLAGS.learning_rate, FLAGS.max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=50)
        merge=tf.merge_all_summaries()

        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        with tf.variable_scope("Model", reuse=True):
            valid_model = model.inference_graph(
                char_vocab_size=char_vocab.size,
                word_vocab_size=word_vocab.size,
                char_embed_size=FLAGS.char_embed_size,
                batch_size=FLAGS.batch_size,
                num_highway_layers=FLAGS.highway_layers,
                num_rnn_layers=FLAGS.rnn_layers,
                rnn_size=FLAGS.rnn_size,
                max_word_length=max_word_length,
                kernels=eval(FLAGS.kernels),
                kernel_features=eval(FLAGS.kernel_features),
                num_unroll_steps=FLAGS.num_unroll_steps,
                dropout=0.0)#dropout to 0 when testing
            valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

        init=tf.initialize_all_variables()#hjq

    with tf.Session(graph=g) as session:#hjq

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)#hjq,graph
        best_valid_loss = None

        if FLAGS.load_model:
            saver.restore(session, FLAGS.load_model)
            print('Loaded model from', FLAGS.load_model, 'saved at global step', train_model.global_step.eval())
        else:
            session.run(init) #hjq
            print('Created and initialized fresh model. Size:', model.model_size())

        ''' take learning rate from CLI, not from saved graph '''
        session.run(
            tf.assign(train_model.learning_rate, FLAGS.learning_rate),
        )

        session.run(train_model.char_embedding)#hjq

        ''' training starts here '''

        # rnn_state = session.run(train_model.initial_rnn_state)
        for epoch in range(FLAGS.max_epochs):
            rnn_state = session.run(train_model.initial_rnn_state)#hjq
            start = time.time()
            avg_train_loss = 0.0
            count = 0
            for x, y in train_reader.iter():
                count += 1
                start_time = time.time()
                # rnn_state,loss, step, grad_norm, _, = session.run([
                #     # sequence order change, hjq
                #     train_model.final_rnn_state,
                #     train_model.loss,
                #     train_model.global_step,
                #     train_model.global_norm,
                #     train_model.train_op,
                loss, _, rnn_state, grad_norm, step = session.run([
                    train_model.loss,
                    train_model.train_op,
                    train_model.final_rnn_state,
                    train_model.global_norm,
                    train_model.global_step,
                ], {
                    train_model.input  : x,
                    train_model.targets: y,
                    train_model.initial_rnn_state: rnn_state
                })

                # clear_char_embedding_padding()
                session.run(train_model.char_embedding)# hjq

                avg_train_loss += 0.05 * (loss - avg_train_loss)

                time_elapsed = time.time() - start_time

                if count % FLAGS.print_every == 0:
                    print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, gradnorm=%6.8f' % (step,
                                                                                                                          epoch, count,
                                                                                                                          train_reader.length,
                                                                                                                          loss, np.exp(loss),
                                                                                                                          time_elapsed,
                                                                                                                          grad_norm))

            # epoch done: time to evaluate
            avg_valid_loss = 0.0
            count = 0
            rnn_state = session.run(valid_model.initial_rnn_state)
            for x, y in valid_reader.iter():
                count += 1
                start_time = time.time()

                loss, rnn_state = session.run([
                    valid_model.loss,
                    valid_model.final_rnn_state
                ], {
                    valid_model.input  : x,
                    valid_model.targets: y,
                    valid_model.initial_rnn_state: rnn_state,
                })

                if count % FLAGS.print_every == 0:
                    print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                avg_valid_loss += loss / valid_reader.length

            print("at the end of epoch:", epoch)
            print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
            print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))
            print("epoch time: %6.4f s"%(time.time()-start))#hjq

            save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_valid_loss)
            saver.save(session, save_as)
            print('Saved model', save_as)

            ''' write out summary events '''
            summary_str =tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss),
                # tf.Summary.Value(tag="train_perplexity")
            ])
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()#hjq,clear
            ''' decide if need to decay learning rate '''
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(train_model.learning_rate)
                print('learning rate was:', current_learning_rate)
                current_learning_rate *= FLAGS.learning_rate_decay
                if current_learning_rate < 1.e-5:
                    print('learning rate too small - stopping now')
                    break

                session.run(train_model.learning_rate.assign(current_learning_rate))
                print('new learning rate is:', current_learning_rate)
            else:
                best_valid_loss = avg_valid_loss


if __name__ == "__main__":
    tf.app.run()
