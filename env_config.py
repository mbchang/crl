import os

from dataloader.mnist_dataset import load_mnist_datasets
from dataloader.mnist_transforms import shrink_mnist_dataset, cuda_mnist_dataset, TransformationCombinationDataLoader
from dataloader.transformation_combination import ConcatTransformationCombiner, TransformationCombiner, SpatialImageTransformations
from log import Logger

def load_image_xforms_env(args, use_cuda, mix_in_normal):
    mnist_orig = load_mnist_datasets('../data', normalize=False)
    mnist_shrunk = shrink_mnist_dataset(mnist_orig, (64, 64))
    if use_cuda: mnist_shrunk = cuda_mnist_dataset(mnist_shrunk)

    kwargs = {}

    transform_config = lambda: SpatialImageTransformations(cuda=use_cuda, **kwargs)

    if mix_in_normal:
        train_combiner=ConcatTransformationCombiner(transformation_combiners=[
            TransformationCombiner(transform_config(), name='3c2_RT', mode='train', cuda=use_cuda),
            TransformationCombiner(transform_config(), name='identity', mode='train', cuda=use_cuda)])
    else:
        train_combiner = TransformationCombiner(transform_config(), name='3c2_RT', mode='train', cuda=use_cuda)

    transformation_combinations = {
        'train': train_combiner,
        'val': TransformationCombiner(transform_config(), name='3c2_RT', mode='val', cuda=use_cuda),
        'test': TransformationCombiner(transform_config(), name='3c3_SRT', mode='test', cuda=use_cuda),
    }

    env = TransformationCombinationDataLoader(
        dataset=mnist_shrunk,
        transformation_combinations=transformation_combinations,
        transform_config=transform_config(),
        cuda=use_cuda)  # although we can imagine not doing this 
    return env

def create_logger(build_expname, args):
    if args.resume:
        """
            - args.resume identifies the checkpoint that we will load the model
            - We will load the args from the saved checkpoint and overwrite the 
            default args.
            - The only things we will not overwrite is args.eval and args.resume,
            which have been provided by the current run
            - We will also set the resumed_from attribute of logger to point to
            the current checkpoint we just loaded up.
        """
        if args.eval:
            logdir = os.path.dirname(args.resume)
            logger = Logger(
                expname='',  # will overwrite
                logdir=logdir,
                params={},  # will overwrite
                resumed_from=args.resume)
            args = logger.load_params_eval(args.eval, args.resume)
            expname = build_expname(args) + '_eval'
            logger.set_expname(expname)
            logger.save_params(logger.logdir, args, ext='_eval')
        elif args.transfer:
            expname = build_expname(args) + '_transfer'
            logger = Logger(
                expname=expname,
                logdir=os.path.join(args.outputdir, expname),
                params=args,
                resumed_from=args.resume)
            logger.save_params(logger.logdir, args, ext='_transfer')
        else:
            assert False, 'You tried to resume but you did not specify whether we are in eval or transfer mode'
    else:
        expname = build_expname(args)   
        logger = Logger(
            expname=expname, 
            logdir=os.path.join(args.outputdir, expname), 
            params=args, 
            resumed_from=None)
        logger.save_params(logger.logdir, args)
    return logger
