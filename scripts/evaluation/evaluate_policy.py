import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        # raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        pass
        # raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        aa = np.random.rand(7)
        aa[...,6:] = 1.
        return aa
        # raise NotImplementedError


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False, rank=0, each_length=1000):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # import ipdb;ipdb.set_trace()
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    # import ipdb;ipdb.set_trace()
    eval_sequences = get_sequences(NUM_SEQUENCES, num_workers=1)
    eval_sequences = eval_sequences[rank*each_length:(rank+1)*each_length]
    orig_eval_sequences = eval_sequences
    print(rank, eval_sequences)
    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    iii = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        iii += 1
        if not os.path.exists(eval_log_dir):
            os.makedirs(eval_log_dir)
        with open(eval_log_dir / 'results_orig_{}.json'.format(rank*each_length+iii), 'w') as file:
            json.dump({'results': result, 'sequences': eval_sequence}, file)    
    with open(eval_log_dir / 'results_orig.json', 'w') as file:
        json.dump({'results': results, 'sequences': eval_sequences}, file)        
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)
    
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    # import ipdb;ipdb.set_trace()
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        # env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        # import ipdb;ipdb.set_trace()
        # (553916, 553961)
        # 'slide right the pink block'
        # ep = np.load('/xxx/xxx/Anonymous/calvin_debug_dataset/validation/episode_' + str(553916 + step).zfill(7) + '.npz', allow_pickle=True)
        action_list = model.step(obs, lang_annotation)
        # action = ep['rel_actions'][:7]
        # action = ep['actions'][:7]
        
        
        # import ipdb;ipdb.set_trace()
        for ii in range(len(action_list)):
            action = action_list[ii]
            obs, _, _, current_info = env.step(action)

            # img = env.render(mode="rgb_array")
            # from PIL import Image
            # Image.fromarray(img).save('temp3_{}_{}.png'.format(lang_annotation, step))
            # obs['rgb_obs']['rgb_static']
            if debug:
                img = env.render(mode="rgb_array")
                from PIL import Image
                Image.fromarray(img).save('temp_{}_{}.png'.format(lang_annotation, step))
                # join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            if step == 0:
                # for tsne plot, only if available
                collect_plan(model, plans, subtask)

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                # if debug:
                print(colored("success", "green"), end=" ")
                return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False

def evaluate_policy_GRMG(
    model,
    env,      
    rank,
    each_length,
):
    # import ipdb;ipdb.set_trace()
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = get_sequences(NUM_SEQUENCES, num_workers = 10)
    # eval_sequences = get_sequences(100, num_workers = 10)
    eval_sequences = eval_sequences[rank*each_length:(rank+1)*each_length] 
    results = []
    sequence_i = 0
    
    for index, (initial_state, eval_sequence) in enumerate(eval_sequences):

        result = eval_sequence_GRMG(env, model, task_oracle, initial_state, eval_sequence, val_annotations, sequence_i)
        results.append(result)
        success_list = count_success(results)
        print(f"{index} done!!!, {result}", flush = True)
    
    import ipdb;ipdb.set_trace()
    
    return results

def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success

def eval_sequence_GRMG(env, model, task_checker, initial_state, eval_sequence, val_annotations, sequence_i):

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout_GRMG(env, model, task_checker, subtask, val_annotations, subtask_i, sequence_i)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout_GRMG(env, model, task_oracle, subtask, val_annotations, subtask_i, sequence_i):

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    for i in range(EP_LEN):

        action_list = model.step(obs, lang_annotation)

        for ii in range(len(action_list)):
            action = torch.tensor(action_list[ii])
            # import ipdb;ipdb.set_trace()
            obs, _, _, current_info = env.step(action)

            # import ipdb;ipdb.set_trace()

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                # import ipdb;ipdb.set_trace()
                print(colored("success", "green"), end=" ", flush = True)
                return True
    print("fail", flush = True)
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, epoch=0., debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()
