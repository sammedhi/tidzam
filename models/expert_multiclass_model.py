import tensorflow as tf
import src.lib as tl

class DNN:
    def __init__(self,conf_data):
        n_classes = len(conf_data["classes_list"])
        data_size = conf_data["size"]
        classes = conf_data["classes"]

        self.name = "selector"
        self.show_kernel_map = []
        self.res = 0.0
        self.model_number = 0.0

        with tf.name_scope('Input'):
            self.input = tf.placeholder(tf.float32, shape=[None, data_size[0] * data_size[1] ], name="x-input")

        with tf.name_scope('Labels'):
            self.labels = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")

        with tf.name_scope('DropOut'):
            self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('model'):
            net = tf.reshape(self.input, shape=[-1, data_size[0], data_size[1], 1])

            with tf.variable_scope("CONV_1"):
                [conv1, W, b] = tl.conv2d(net, 121, 20)
                R1 = tf.nn.l2_loss(W)
                self.show_kernel_map.append(W) # Create the feature map

            with tf.variable_scope("POOL_1"):
                pool1 = tl.max_pool_2x2(conv1)

            with tf.variable_scope("CONV_2"):
                [conv2, W, b] = tl.conv2d(pool1, 16, 10)
                R2 = tf.nn.l2_loss(W)
                self.show_kernel_map.append(W) # Create the feature map

            with tf.variable_scope("POOL_2"):
                pool2 = tl.max_pool_2x2(conv2)

            self.out = self.build_expert_fc(classes , pool2)

            with tf.name_scope('Cost'):
                self.cost  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits (
                    labels=self.labels,
                    logits=self.out) )

                self.cost = self.cost + 0.01 * (R1 + R2 + self.res) / self.model_number

            self.output = tf.nn.sigmoid(self.out)

    def build_expert_fc(self , classes , fc_input , node_name = ""):
        self.model_number += 1.0
        output_size = len(classes)
        sub_outputs = []
        for index , cl in enumerate(classes):
            if not isinstance(cl , str):
                sub_outputs.append({"outputs" : self.build_expert_fc(cl["classes"] , fc_input , node_name + "_" + cl["name"]) ,
                                    "parent_index" : index})

        with tf.variable_scope("Expert_" + node_name):
            with tf.variable_scope("FC_1"):
                flat1 = tl.fc_flat(fc_input)
                h, W, b =  tl.fc(flat1, 1024)
                self.res += tf.nn.l2_loss(W)
                fc1   = tf.nn.relu(h)

            with tf.variable_scope("DROPOUT_1"):
                drop1 = tf.nn.dropout(fc1, self.keep_prob)

            with tf.variable_scope("FC_2"):
                h, W, b =  tl.fc(drop1, 1024)
                self.res += tf.nn.l2_loss(W)
                fc2   = tf.nn.relu( h )

            with tf.variable_scope("DROPOUT_2"):
                drop2 = tf.nn.dropout(fc2, self.keep_prob)

            with tf.variable_scope("OUT"):
                out, W, b = tl.fc(drop2, output_size)

        with tf.variable_scope("WeightingOp"):
            offset = 0
            for sub_output in sub_outputs:
                factors_list = []
                for t in range(sub_output["outputs"].shape[1]):
                    factors_list.append( out[: , sub_output["parent_index"] + offset])
                tf_factor = tf.stack( factors_list , axis = 1 )
                new_sub_output = - tf.reduce_logsumexp (tf.stack( (- sub_output["outputs"] ,
                                                                   - tf_factor ,
                                                                   - tf.add(sub_output["outputs"] ,
                                                                   tf_factor)  )   , axis = 2)  , axis = 2 )

                out = tf.concat( (out[ : , : sub_output["parent_index"] + 1 + offset] ,
                                  new_sub_output ,
                                  out[ : ,  sub_output["parent_index"] + 1 + offset: ]) , 1)

                offset += sub_output["outputs"].shape[1]

        return out


'''
    def build_expert_fc(self , node , input , tf_factor = None):
        if len(node.child_list) == 0:
            return None

        self.model_number += 1.0
        with tf.variable_scope("Expert_" + node.name):
            with tf.variable_scope("FC_1"):
                flat1 = tl.fc_flat(input)
                h, W, b =  tl.fc(flat1, 1024)
                self.res += tf.nn.l2_loss(W)
                fc1   = tf.nn.relu(h)

            with tf.variable_scope("DROPOUT_1"):
                drop1 = tf.nn.dropout(fc1, self.keep_prob)

            with tf.variable_scope("FC_2"):
                h, W, b =  tl.fc(drop1, 1024)
                self.res += tf.nn.l2_loss(W)
                fc2   = tf.nn.relu( h )

            with tf.variable_scope("DROPOUT_2"):
                drop2 = tf.nn.dropout(fc2, self.keep_prob)

            with tf.variable_scope("OUT"):
                out, W, b = tl.fc(drop2, len (node.child_list))

            if tf_factor is not None:
                with tf.variable_scope("WeightingOp"):
                    factors_list = []
                    for t in range(len(node.child_list)):
                        factors_list.append(tf_factor)
                    tf_factor = tf.stack( factors_list , axis = 1 )
                    out = - tf.reduce_logsumexp (tf.stack( (- out , - tf_factor , - tf.add(out , tf_factor) , tf.zeros_like(out) ), axis = 2)  , axis = 2 )

        for i , child in enumerate(node.child_list):
            new_expert = self.build_expert_fc(child , input , out[:,i])
            if new_expert is not None:
                with tf.variable_scope("Concatenate_Op"):
                    out = tf.concat( (out , new_expert) , 1)

        return out
'''


'''
            with tf.variable_scope("FC_1"):
                flat1 = tl.fc_flat(pool2)
                h, W, b =  tl.fc(flat1, 1024)
                R3 = tf.nn.l2_loss(W)
                fc1   = tf.nn.relu(h)

            with tf.variable_scope("DROPOUT_1"):
                drop1 = tf.nn.dropout(fc1, self.keep_prob)

            with tf.variable_scope("FC_2"):
                h, W, b =  tl.fc(drop1, 1024)
                R4 = tf.nn.l2_loss(W)
                fc2   = tf.nn.relu( h )

            with tf.variable_scope("DROPOUT_2"):
                drop2 = tf.nn.dropout(fc2, self.keep_prob)

            with tf.variable_scope("OUT"):
                self.out, W, b = tl.fc(drop2, n_classes)

        with tf.name_scope('Cost'):
            self.cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels,
                logits=self.out) )

            self.cost = self.cost + 0.01 * (R1 + R2 + R3 + R4)

'''
