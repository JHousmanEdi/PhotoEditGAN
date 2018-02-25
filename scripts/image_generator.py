from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import dataprocessing as ds
import cgan_model as gp
from argsource import args
import tensorflow as tf
import numpy as np
import random
import os
import json
import time
import math
import base64


def export_generator():
    input_src = tf.placeholder(tf.string, shape=[1])
    input_data = tf.decode_base64(input_src[0])
    input_image = tf.image.decode_png(input_data)
    input_image = input_image[:, :, :3]
    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
    input_image.set_shape([args['cropping'], args['cropping'], 3])
    batch_input = tf.expand_dims(input_image, axis=0)

    with tf.variable_scope('generator') as scope:
        generator = gp.Generator(ds.preprocess(batch_input), 3, args['ngf'])
        batch_output = ds.deprocess(generator.build())

    output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]

    output_data = tf.image.encode_jpeg(output_image, quality=100)

    output = tf.convert_to_tensor([tf.encode_base64(output_data)])
    key = tf.placeholder(tf.string, shape=[1])
    inputs = {
        "key": key.name,
        "input": input_src.name
    }
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {
        "key": tf.identity(key).name,
        "output": output.name
    }
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("loading model")  # Loading previous model parameters
        checkpoint = tf.train.latest_checkpoint(args['checkpoint'])
        restore_saver.restore(sess, checkpoint)
        print("exporting model")
        export_saver.export_meta_graph(filename=os.path.join(args['results_dir'], "export.meta"))
        export_saver.save(sess, os.path.join(args['results_dir'], "export"), write_meta_graph=False)


def generate_image(image_name):
    image_directory = os.path.join(os.getcwd(), "Dataset", "GenerateMe")
    input_image = os.path.join(image_directory, image_name)
    with open(input_image, "rb") as f:
        input_data = f.read()
    input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
    input_instance = json.loads(json.dumps(input_instance))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args['stand_alone'] + "/export.meta")
        saver.restore(sess,args['stand_alone'] + "/exports")
        input = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
        output = tf.get_default_graph().get_tensor_by_name('packed:0')
        input_value = np.array(input_instance["input"])
        output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]

    output_instance = dict(output=output_value.decode("ascii"), key="0")
    b64data = output_instance["output"]
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))
    out_image_name = os.path.splitext(input_image)[0]
    output_image = os.path.join(image_directory, out_image_name + "_out.jpg")
    with open(output_image, "wb") as f:
        f.write(output_data)
        f.close()


def main():
    # noinspection PyUnresolvedReferences
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if args['seed'] is None:
        args['seed'] = random.randint(0, 2**31 - 1)

    tf.set_random_seed(args['seed'])
    # noinspection PyUnresolvedReferences
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    if not os.path.exists(args['results_dir']):
        os.makedirs(args['results_dir'])

    if args['mode'] == "test" or args['mode'] == "export":
        if args['checkpoint'] is None:
            raise Exception("Checkpoint required for test mode")

        options = {"ngf", "ndf"}
        with open(os.path.join(args['checkpoint'],  "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("Resuming with {} = {}".format(key, val))
                    args[key] = val
        args['scaling'] = args['cropping']
        args['flip'] = False

    for key, val in args.items():
        print("{} = {}".format(key, val))

    with open(os.path.join(args['log_dir'], "options.json"), "w") as f:
        f.write(json.dumps(args, sort_keys=True, indent=4))

    if args['mode'] == "export":
        export_generator()

        return

    if args['mode'] == "generate":
        if args['image_name'] is None:
            raise Exception("Please cannot proceed without an image")
        else:
            if type(args['image_name']) is list:
                for i in args['image_name']:
                    generate_image(i)
            else:
                generate_image(args['image_name'])
        return

    examples = ds.image_loader()
    print("examples count = %d" % examples.count)

    model = gp.Model(examples.inputs, examples.targets, args['ngf'], args['ndf'], 3)
    model_out = model.optimize()
    inputs = ds.deprocess(examples.inputs)
    targets = ds.deprocess(examples.targets)
    outputs = ds.deprocess(model_out.outputs)

    with tf.name_scope("convert_input_images"):
        converted_input_images = ds.convert(inputs)
    with tf.name_scope("convert_retouched_images"):
        converted_target_images = ds.convert(targets)
    with tf.name_scope("convert_outputs_images"):
        converted_output_images = ds.convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_jpeg, converted_input_images, dtype=tf.string, name="input_jpgs"),
            "targets": tf.map_fn(tf.image.encode_jpeg, converted_target_images, dtype=tf.string, name="target_jpgs"),
            "outputs": tf.map_fn(tf.image.encode_jpeg, converted_output_images, dtype=tf.string, name="output_jpgs"),
        }

    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_input_images)
    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_target_images)
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_output_images)
    with tf.name_scope("real_discriminator_summary"):
        tf.summary.image("real_discriminator", tf.image.convert_image_dtype(model_out.predict_real, dtype=tf.uint8))
    with tf.name_scope("fake_discriminator_summary"):
        tf.summary.image("fake_discriminator",  tf.image.convert_image_dtype(model_out.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model_out.D_loss)
    tf.summary.scalar("encoder_decoder_loss", model_out.G_loss)
    tf.summary.scalar("generator_l2_loss", model_out.gen_l2_loss)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)  #Summary of results

    for grad, var in model_out.discrim_grad_vars + model_out.gen_grads_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    if args['trace_freq'] > 0 or args['summary_freq'] > 0:
        logdir = args['log_dir']
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            print("parameter_count=", sess.run(parameter_count))

            if args['checkpoint'] is not None:
                print("Continuing from last checkpoint")
                checkpoint = tf.train.latest_checkpoint(args['log_dir'])
                saver.restore(sess, checkpoint)

            max_steps = 2**32
            if args['max_epochs'] is not None:
                max_steps = examples.steps_per_epoch * args['max_epochs']
            if args['max_steps'] is not None:
                max_steps = args['max_steps']

            if args['mode'] == "test":

                max_steps = min(examples.steps_per_epoch, max_steps)
                for step in range(max_steps):
                    results = sess.run(display_fetches)
                    filesets = ds.save_image(results, "test")
                    index_path = ds.results_to_html(filesets)
                    for i, f in enumerate(filesets):
                        print("evaluated image", f["name"])
                    print("wrote index at", index_path)
            else:
                start = time.time()

                for step in range(max_steps):
                    def right_time(freq):
                        return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                    options = None
                    run_metadata = None
                    if right_time(args['trace_freq']):
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                    fetches = {
                        "train": model_out.train,
                        "global_step": sv.global_step,
                    }

                    if right_time(args['progress_freq']):
                        fetches["discriminator_loss"] = model_out.D_loss
                        fetches["encoder_decoder_loss"] = model_out.G_loss
                        fetches["generator_l2_loss"] = model_out.gen_l2_loss

                    if right_time(args['summary_freq']):
                        fetches["summary"] = sv.summary_op

                    if right_time(args['display_freq']):
                        fetches["display"] = display_fetches

                    results = sess.run(fetches, options=options, run_metadata=run_metadata)

                    if right_time(args['trace_freq']):
                        print("Recording current stack trace")
                        sv.summary_writer.add_run_metadata(run_metadata, "step_{}".format(results["global_step"]))

                    if right_time(args['progress_freq']):
                        train_epoch = math.ceil(results["global_step"]/examples.steps_per_epoch)
                        train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                        rate = (step + 1) * args['batch'] / (time.time() - start)
                        remaining = (max_steps - step) * args['batch'] / rate
                        print("progress epoch: {} step: {} image/sec: {} remaining: {}".format(train_epoch,
                                                                                               train_step,
                                                                                               rate, remaining))
                        print("discriminator_loss: {}".format(results["discriminator_loss"]))
                        print("encoder_decoder_loss: {}".format(results["encoder_decoder_loss"]))
                        print("generator_l2_loss {}".format(results['generator_l2_loss']))
                    if right_time(args['summary_freq']):
                        print("Recording current summary of training")
                        sv.summary_writer.add_summary(results["summary"], results["global_step"])

                    if right_time(args['display_freq']):
                        print("Saving progress images")
                        filesets = ds.save_image(results["display"], 'ProgExpE', step=results["global_step"])
                        ds.results_to_html(filesets, step=True)

                    if right_time(args['save_freq']):
                        print("saving current model parameters")
                        saver.save(sess, os.path.join(args['initial_ckpt'], "model"), global_step=sv.global_step)

                    if sv.should_stop():
                        break


main()



















