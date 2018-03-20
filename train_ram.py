import utils
from utils import Plotter
import numpy as np
import tensorflow as tf
import time
import argparse

NUM_CLASSES = 10

BATCH_SIZE = 25
SAMPLE_M = 10

NUM_GLIMPSES = 6
STD_DEV = .17
STATE_SIZE = 256

NUM_EPOCHS = 300
LEARNING_DECAY = .975

DISPLAY_INTERVAL = 400
NUM_EXAMPLES = 4

TRANSLATE_DIM = 60

M_BATCH_SIZE = BATCH_SIZE * SAMPLE_M

parser = argparse.ArgumentParser(description='Train a recurrent attention model (RAM) in Tensorflow')
parser.add_argument("--translate", action="store_true", help="use the MNIST-translate dataset")

def fc(x, scope, nh):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable('w', [nin, nh], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable('b', [nh], initializer=tf.constant_initializer(0))

    return tf.matmul(x, w)+b

def glimpse_network(glimpses, locs):
    with tf.variable_scope('glimpse_network'):
        h1 = tf.nn.relu(fc(glimpses, 'h1', nh=128))
        h2 = tf.nn.relu(fc(locs, 'h2', nh=128))
        out = tf.nn.relu(fc(tf.concat([h1, h2], axis=1), 'glimpse_out', nh=256))

    return out

def core_network(gs, states):
    next_states = tf.nn.relu(fc(tf.concat([gs, states], axis=1), 'core_network', nh=STATE_SIZE))
    return next_states

def location_network(states):
    next_locs = fc(states, 'location_network', nh=2)
    return next_locs

def action_network(states):
    next_actions = tf.nn.relu(fc(states, 'action_network', nh=NUM_CLASSES))
    return next_actions

def baseline_network(states):
    baselines = fc(states, 'baseline_network', nh=1)
    return baselines

def visualize_glimpse_slice(img, loc, g_dim, color_idx, color_weight):
    img_dim = np.shape(img)[1]

    start = np.round(((loc + 1) / 2) * img_dim - g_dim / 2)
    start = np.ndarray.astype(start, np.int32)

    x1, x2 = max(start[0], 0), min(start[0]+g_dim, img_dim)
    y1, y2 = max(start[1], 0), min(start[1]+g_dim, img_dim)

    glimpse = img[x1:x2,y1:y2,:]
    glimpse[:,:,color_idx] = color_weight + (1 - color_weight) * glimpse[:,:,color_idx]

    img[x1:x2,y1:y2,:] = glimpse

def main():
    args = parser.parse_args()
    
    if args.translate:
        GLIMPSE_DIM = 12
        NUM_SCALES = 3

        train_data, train_labels, test_data, test_labels = utils.create_translated_mnist(TRANSLATE_DIM)
    else:
        GLIMPSE_DIM = 8
        NUM_SCALES = 1

        train_data, train_labels, test_data, test_labels = utils.load_MNIST()

    plotter = Plotter(rows=NUM_EXAMPLES, columns=NUM_GLIMPSES)

    img_dim = np.shape(train_data)[1]

    global_step = tf.Variable(0, name='global_step', trainable=False)
    batch_data = tf.placeholder(tf.float32, shape=[M_BATCH_SIZE, img_dim, img_dim, 1])
    batch_labels = tf.placeholder(tf.int32, shape=[M_BATCH_SIZE])
    learn_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        states = tf.zeros([M_BATCH_SIZE, STATE_SIZE], dtype=tf.float32)
        locs = tf.random_uniform([M_BATCH_SIZE, 2], minval=-1, maxval=1)

        state_list = []
        loc_list = []
        logp_list = []

        for j in range(NUM_GLIMPSES):
            loc_list.append(locs)
            glimpse_list = []

            for k in range(NUM_SCALES):
                if k > 0:
                    new_dim = img_dim // (2 ** k)
                    resized_batch = tf.image.resize_images(batch_data, [new_dim, new_dim])
                else:
                    resized_batch = batch_data

                glimpses = tf.image.extract_glimpse(resized_batch, [GLIMPSE_DIM, GLIMPSE_DIM], locs)
                glimpses = tf.reshape(glimpses, [glimpses.get_shape()[0].value, -1])
                glimpse_list.append(glimpses)

            glimpse_data = tf.concat(glimpse_list, axis=1)

            gs = glimpse_network(glimpse_data, locs)
            states = core_network(gs, states)
            mean_locs = location_network(states)

            state_list.append(states)

            gaussian = tf.contrib.distributions.Normal(mean_locs, STD_DEV)

            sampled_locs = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: gaussian.sample(), lambda: mean_locs)
            sampled_locs = tf.stop_gradient(sampled_locs)
            locs = tf.clip_by_value(sampled_locs, -1, 1)
            
            logp = gaussian.log_prob(sampled_locs)
            logp = tf.reduce_sum(logp, axis=1)
            logp_list.append(logp)

        all_states = tf.reshape(tf.transpose(tf.stack(state_list, axis=0), [1, 0, 2]), [M_BATCH_SIZE * NUM_GLIMPSES, -1])

        action_logits = action_network(states)
        actions = tf.argmax(action_logits, axis=1, output_type=tf.int32)
        rewards = tf.cast(tf.equal(actions, batch_labels), tf.float32)
        rewards = tf.tile(tf.reshape(rewards, [M_BATCH_SIZE, 1]), [1, NUM_GLIMPSES])
        rewards = tf.reshape(rewards, [M_BATCH_SIZE * NUM_GLIMPSES, 1])

        all_action_logits = action_network(all_states)
        all_actions = tf.argmax(all_action_logits, axis=1, output_type=tf.int32)

        baselines = baseline_network(all_states)
        baselines_loss = tf.reduce_mean(tf.square(baselines - rewards))

        logp = tf.stack(logp_list)
        logp = tf.transpose(logp)

        loc_loss = -1 * tf.multiply(logp, tf.reshape(tf.stop_gradient(rewards - baselines), [M_BATCH_SIZE, NUM_GLIMPSES]))
        loc_loss = tf.reduce_mean(loc_loss)

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=batch_labels)
        class_loss = tf.reduce_mean(class_loss)

        tr_vars = tf.trainable_variables()

        loc_vars = [var for var in tr_vars if 'location_network' in var.name]
        baseline_vars = [var for var in tr_vars if 'baseline_network' in var.name]
        class_vars = [var for var in tr_vars if ('core_network' in var.name) or ('glimpse_network' in var.name) or ('action_network' in var.name)]

        opt_CL = tf.train.AdamOptimizer(learn_rate).minimize(class_loss, var_list=class_vars)
        opt_L = tf.train.AdamOptimizer(learn_rate).minimize(loc_loss, var_list=loc_vars)
        opt_B = tf.train.AdamOptimizer(learn_rate).minimize(baselines_loss, var_list=baseline_vars)

        opts = [opt_CL, opt_L, opt_B]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    num_images = len(train_data)
    num_iterations = num_images // BATCH_SIZE

    num_test = len(test_data)
    num_test_iterations = num_test // M_BATCH_SIZE

    start = time.time()

    for epoch in range(sess.run(global_step), NUM_EPOCHS):
        print('__EPOCH_{0}__'.format(epoch))
        perm = np.random.permutation(num_images)

        shuffled_images = train_data[perm]
        shuffled_labels = train_labels[perm]

        accs = []
        losses = []

        _learn_rate = .001 * (LEARNING_DECAY ** epoch)

        for i in range(num_iterations):
            offset = (i * BATCH_SIZE) % num_images
            _batch_data = shuffled_images[offset:(offset + BATCH_SIZE)]
            _batch_labels = shuffled_labels[offset:(offset + BATCH_SIZE)]

            _batch_data = np.tile(_batch_data, [SAMPLE_M, 1, 1, 1])
            _batch_labels = np.tile(_batch_labels, SAMPLE_M)

            feed_dict = {batch_data: _batch_data, batch_labels: _batch_labels, learn_rate: _learn_rate, is_training: True}

            _, _action_logits, _losses = sess.run([opts, action_logits, [class_loss, loc_loss, baselines_loss]], feed_dict=feed_dict)

            out_labels = np.argmax(_action_logits, axis=1)

            acc = np.sum(out_labels == _batch_labels) / M_BATCH_SIZE
            accs.append(acc)

            losses.append(_losses)

            if i % DISPLAY_INTERVAL == 0:
                _loc_list, _all_actions = sess.run([loc_list, all_actions], feed_dict=feed_dict)

                print('Batch {0}/{1}'.format(i, num_iterations))
                print('  (Batch losses) Cross entropy: {0:<8.3g} Policy: {1:<8.3g} Baseline: {2:<8.3g}'.format(*_losses))

                example_data = []

                for img_num in range(NUM_EXAMPLES):
                    img = _batch_data[img_num]
                    img = np.tile(img, (1, 1, 3))

                    for t in range(NUM_GLIMPSES):
                        action = _all_actions[img_num * NUM_GLIMPSES + t]
                        is_correct = action == _batch_labels[img_num]
                        color_idx = 1 if is_correct == 1 else 0

                        loc = _loc_list[t][img_num]

                        new_img = np.copy(img)
                        color_weight = .8 if NUM_SCALES == 1 else .5

                        for k in reversed(range(NUM_SCALES)):
                            visualize_glimpse_slice(new_img, loc, GLIMPSE_DIM * (2 ** k), color_idx, color_weight)

                        example_data.append(new_img)

                plotter.plot(example_data)

        test_accs = []

        for i in range(num_test_iterations):
            offset = (i * M_BATCH_SIZE) % num_images
            _batch_data = test_data[offset:(offset + M_BATCH_SIZE)]
            _batch_labels = test_labels[offset:(offset + M_BATCH_SIZE)]

            feed_dict = {batch_data: _batch_data, batch_labels: _batch_labels, is_training: False}
            _action_logits = sess.run(action_logits, feed_dict=feed_dict)
            out_labels = np.argmax(_action_logits, axis=1)
            acc = np.sum(out_labels == _batch_labels) / M_BATCH_SIZE

            test_accs.append(acc)

        now = time.time()
        elapsed = now - start

        mean_losses = np.mean(losses, axis=0)

        print('Epoch Summary:')
        print('  Learn rate:', _learn_rate)
        print('  Mean train accuracy:', np.mean(accs))
        print('  Mean test accuracy:', np.mean(test_accs))
        print('  (Mean losses) Cross entropy: {0:<8.3g} Policy: {1:<8.3g} Baseline: {2:<8.3g}'.format(*mean_losses))
        print('  Elapsed (since start):', elapsed)

if __name__ == "__main__":
    main()