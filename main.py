import logging
import argparse
import os
import json

from simpletransformers.seq2seq import Seq2SeqModel
from utils import read_data_source_target


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True, help="Input data dir. {train/valid/test}.json files for the task.")
    parser.add_argument("--model_type", default='gpt2', type=str, help="lm type")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="lm name or path")
    parser.add_argument("--init_weights", action="store_true", help="whether fresh init the weights of the model")
    parser.add_argument("--add_tokens", action="store_true", help="whether add the tokens in vocab.json in data_dir to the vocabulary.")
    parser.add_argument("--no_dropout", action="store_true", help="Whether disable dropout.")
    parser.add_argument("--n_layer", default=None, type=int, help="number of layers, only used when init weight")
    parser.add_argument("--n_head", default=None, type=int, help="number of heads, only used when init weight")
    parser.add_argument("--n_inner", default=None, type=int, help="inner dimension of MLP")
    parser.add_argument("--no_ln", action="store_true", help="Whether disable layernorm.")
    parser.add_argument("--no_mlp", action="store_true", help="Whether disable mlp layers.")
    parser.add_argument("--share_mlp", action="store_true", help="Whether share mlp weights across layers.")
    parser.add_argument("--add_recurrence", action="store_true", help="Whether run the layers twice.")
    parser.add_argument("--re_embed", action="store_true", help="Whether add re-embedding during recurrence.")
    parser.add_argument("--re_embed_temp", default=1.0, type=float, help="softmax temperature for re-embedding")
    parser.add_argument("--relation_mean_shift", action="store_true", help="Whether perform OOD relation mean shift w.r.t. ID relations in lm_head")
    parser.add_argument("--add_memory", action="store_true", help="Whether add shared mlp memory.")
    parser.add_argument("--memory_dim", default=1536, type=int, help="inner dimension of add shared mlp memory")

    parser.add_argument("--fp16", action="store_true", help="whether use half-precision training")
    parser.add_argument("--do_train", action="store_true", help="Whether run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether run validation.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Whether to overwrite on the existing output dir")
    parser.add_argument("--save_best_model", action="store_true", help="Whether to save the best model on validation")
    parser.add_argument("--use_multiprocessed_decoding", action="store_true", help="Whether to use multiprocess when decoding")
    parser.add_argument("--save_model_every_epoch", action="store_true", help="Whether to save model every epoch")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Whether to eval model during training")
    parser.add_argument("--predict_during_training", action="store_true", help="Whether to predict on test set during training")

    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--warmup_steps", default=2000, type=int, help="Warmup step. 0 for using warmup ratio.")
    parser.add_argument("--save_epoch_interval", default=0, type=int, help="Save checkpoint every X epochs. 0 for no saving")
    parser.add_argument("--scheduler", default='linear_schedule_with_warmup', type=str, help="scheduler type")
    parser.add_argument("--output_dir", default='output_dir/', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--prediction_dir", default=None, type=str, help="The output directory where the predictions results will be written.")
    parser.add_argument("--custom_test", default=None, type=str, help="Override the default test set (test.json)")
    parser.add_argument("--save_step", default=0, type=int, help="Save checkpoint every X updates steps. 0 for no saving")
    parser.add_argument("--save_step_dense", default=-1, type=int, help="If not -1, save via every save_step_dense_interval steps till specified")
    parser.add_argument("--save_step_dense_interval", default=2000, type=int, help="")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Size of each train batch")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Size of each eval/predict batch")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--learning_rate", default=4e-5, type=float, help="learning rate")
    parser.add_argument("--max_steps", default=0, type=int, help="Number of train steps")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Number of train epochs")
    parser.add_argument('--dataloader_num_workers', default=0, type=int, help='the number of cpus used in collecting data in dataloader. Note that if it is large than cpu number, the program may be stuck')
    parser.add_argument('--manual_seed', default=42, type=int, help='random seed')
    parser.add_argument("--max_seq_length", default=None, type=int, help="Max input seq length")
    parser.add_argument("--max_length", default=None, type=int, help="Max output seq length")
    parser.add_argument("--max_gen_length", default=None, type=int, help="Max seq length appending during generation")
    parser.add_argument("--block_size", default=None, type=int, help="block size")
    parser.add_argument("--prediction_cutoff", default=None, type=int, help="if set, only predict on the first # of prediction examples")
    
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int)

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    train_sample_size = None
    if args.do_train:
        train_df, train_sample_size = read_data_source_target(os.path.join(args.data_dir, "train.json"), return_num=True)
    else:
        train_df = None

    if args.do_eval or args.evaluate_during_training:
        eval_df = read_data_source_target(os.path.join(args.data_dir, "valid.json"))
    else:
        eval_df = None

    if args.do_predict or args.predict_during_training:
        if args.custom_test:
            test_df = read_data_source_target(os.path.join(args.data_dir, args.custom_test), return_json=True)
        else:
            test_df = read_data_source_target(os.path.join(args.data_dir, "test.json"), return_json=True)
    else:
        test_df = None

    new_tokens = None
    if args.add_tokens:
        with open(os.path.join(args.data_dir, "vocab.json")) as f:
            new_tokens = json.load(f)

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": args.overwrite_output_dir,
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_length,
        "max_gen_length": args.max_gen_length,
        "block_size": args.block_size,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_steps": args.save_step,
        "use_multiprocessing": False,
        "output_dir": args.output_dir,
        "manual_seed": args.manual_seed,
        "fp16": args.fp16,
        "truncation": True,
        "dataloader_num_workers":args.dataloader_num_workers,
        "use_multiprocessed_decoding":args.use_multiprocessed_decoding,
        "save_best_model": args.save_best_model,
        "save_model_every_epoch": args.save_model_every_epoch,
        "save_epoch_interval": args.save_epoch_interval,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "evaluate_during_training": args.evaluate_during_training,
        "predict_during_training": args.predict_during_training,
        "mlm": False,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "n_layer": args.n_layer,
        "n_inner": args.n_inner,
        "n_head": args.n_head,
        "memory_dim": args.memory_dim,
    }

    ddp_args = {
        "local_rank": args.local_rank,
        "rank": args.rank,
        "gpu": args.gpu,
        "world_size": args.world_size,
        "dist_url": args.dist_url,
        "dist_backend": args.dist_backend,
    }

    # Initialize model
    model = Seq2SeqModel(
        model_type=args.model_type,
        model_name=args.model_name_or_path,
        args=model_args,
        ddp_args=ddp_args,
        new_tokens=new_tokens,
        init_weights=args.init_weights,
        no_dropout=args.no_dropout,
        no_ln=args.no_ln,
        no_mlp=args.no_mlp,
        share_mlp=args.share_mlp,
        add_memory=args.add_memory,
        add_recurrence=args.add_recurrence,
        re_embed=args.re_embed,
        re_embed_temp=args.re_embed_temp,
        relation_mean_shift=args.relation_mean_shift,
    )

    # Train the model
    if args.do_train:
        model.train_model(train_data=train_df, eval_data=eval_df, test_data=test_df, output_dir=args.output_dir,
                          save_step_dense=args.save_step_dense, save_step_dense_interval=args.save_step_dense_interval)

    # Evaluate the model
    if args.do_eval:
        results = model.eval_model(eval_data=eval_df)

    # Use the model for prediction
    if args.do_predict:
        if args.custom_test:
            model.predict(pred_data=test_df, output_dir=args.prediction_dir, cutoff=args.prediction_cutoff, out_file=args.custom_test)
        else:
            model.predict(pred_data=test_df, output_dir=args.prediction_dir, cutoff=args.prediction_cutoff)


if __name__ == '__main__':
    main()