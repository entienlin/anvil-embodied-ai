"""Standalone real-hardware eval loop for a plain (non-HIL-SERL) policy --
built for the ACT proof-of-concept (2026-09-29), but generic to any
lerobot policy type make_policy() supports.

Why this isn't just hilserl_actor with a different --policy.type: actor.py's
act_with_policy() (see run_actor.py) is built around SAC's online-RL loop --
gRPC queues to a learner, a replay buffer, per-step transition packaging for
training. ACT needs none of that; it's a plain trained policy being
evaluated. This mirrors act_with_policy()'s core env-stepping logic (same
make_robot_env()/make_processors()/make_policy() calls, same
step_env_and_process_transition() helper -- all reused directly, not
reimplemented) but drops everything learner-related.

Also unlike hilserl_gym_manipulator's own control_loop() (record mode):
that loop always sends a neutral (zero) action and relies entirely on human
teleop for real motion -- it has no "let a policy drive" path at all. This
script is the one that actually calls policy.select_action() every tick.

The leader arm / space-bar intervention still works exactly as in every
other HIL-SERL entrypoint: the SAME InterventionActionProcessorStep swaps
in the human's action when intervening, so it remains a live safety
override and episode-success/failure labeling mechanism during eval, not
something this script has to reimplement.

Usage:
    ros2 run lerobot_control hilserl_eval_policy --config_path <path>

Episode count is controlled the same way as hilserl_actor:
HILSERL_MAX_EPISODES=N stops cleanly after N episodes (reusing
patches._request_stop(), which also makes the trailing reset for the
would-be next episode a no-op -- see patches.py's patch_leader_skip_reset_
when_stopping()). Unset/0 runs until Ctrl+C.
"""

import logging
import os

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def _max_episodes() -> int:
    try:
        return int(os.environ.get("HILSERL_MAX_EPISODES", "0"))
    except ValueError:
        return 0


def load_policy_processors(pretrained_path, device):
    """Load the checkpoint's saved pre/post-processor pipelines, exactly the
    way production inference_node.py does (ModelLoader.load_with_processors())
    -- NOT something make_policy()/select_action() apply automatically.

    Found the hard way (2026-09-30): a real-hardware ACT run went straight
    to a wrong, confident target instead of the real task. Checked the raw
    model output directly against the known-correct target for that exact
    input: it was sitting in roughly [-1, 1] (normalized space), not real
    joint degrees. select_action() on the bare policy object never applied
    the checkpoint's own MEAN_STD normalization/denormalization -- for the
    EE-delta version this went unnoticed because raw normalized output
    happens to look like a plausible small EE-delta by coincidence; for
    joint-space degrees the mismatch is dramatic and immediately visible.

    Returns (preprocessor, postprocessor), either None if the checkpoint
    has no policy_preprocessor.json/policy_postprocessor.json (a real,
    loud problem for a MEAN_STD-normalized checkpoint like ours, not a
    silently-fine default -- callers should treat None as "inference will
    be wrong", not "no processing needed").
    """
    import importlib
    from pathlib import Path

    from lerobot.processor import PolicyProcessorPipeline

    path = Path(pretrained_path)
    pre_processor = post_processor = None

    try:
        importlib.import_module("lerobot.policies.act.processor_act")
    except ModuleNotFoundError:
        pass

    if (path / "policy_preprocessor.json").exists():
        pre_processor = PolicyProcessorPipeline.from_pretrained(str(path), config_filename="policy_preprocessor.json")
        if hasattr(pre_processor, "to"):
            pre_processor.to(device)
    if (path / "policy_postprocessor.json").exists():
        post_processor = PolicyProcessorPipeline.from_pretrained(str(path), config_filename="policy_postprocessor.json")
        if hasattr(post_processor, "to"):
            post_processor.to(device)

    if pre_processor is None and post_processor is None:
        print(
            f"[WARNING] no policy_preprocessor.json/policy_postprocessor.json found in {path} -- "
            "observations/actions will NOT be normalized. If this checkpoint was trained with "
            "MEAN_STD normalization (ours are), inference output will be wrong.",
            flush=True,
        )
    return pre_processor, post_processor


def run_policy_eval(cfg) -> None:
    import torch
    from lerobot.policies.factory import make_policy
    from lerobot.rl.gym_manipulator import (
        create_transition,
        make_processors,
        make_robot_env,
        step_env_and_process_transition,
    )
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.types import TransitionKey
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.random_utils import set_seed

    from lerobot_control.hilserl import patches

    logging.info("[EVAL] make_env")
    env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    get_safe_torch_device(cfg.policy.device, log=True)

    logging.info("[EVAL] make_policy")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy = policy.eval()
    pre_processor, post_processor = load_policy_processors(cfg.policy.pretrained_path, cfg.policy.device)

    input_keys = list(cfg.policy.input_features)

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    policy.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    n_target = _max_episodes()
    episode_idx = 0
    sum_reward = 0.0
    episode_steps = 0
    episode_intervention_steps = 0

    print(
        f"[EVAL] starting -- policy={cfg.policy.type}, "
        f"{'stopping after ' + str(n_target) + ' episodes' if n_target > 0 else 'running until Ctrl+C'}",
        flush=True,
    )

    try:
        while True:
            observation = {k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in input_keys}
            if pre_processor is not None:
                observation = pre_processor(observation)
            with torch.no_grad():
                action = policy.select_action(batch=observation)
            if post_processor is not None:
                action = post_processor.process_action(action)

            new_transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            reward = new_transition[TransitionKey.REWARD]
            sum_reward += float(reward)
            episode_steps += 1

            if new_transition[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False):
                episode_intervention_steps += 1

            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)
            transition = new_transition

            if done or truncated:
                rate = episode_intervention_steps / episode_steps if episode_steps else 0.0
                logging.info(
                    f"[EVAL-STATS] episode={episode_idx} reward={sum_reward} "
                    f"steps={episode_steps} intervention_rate={rate:.3f}"
                )
                print(
                    f"[EVAL] episode {episode_idx} done: reward={sum_reward} "
                    f"steps={episode_steps} intervention_rate={rate:.3f}",
                    flush=True,
                )
                episode_idx += 1
                sum_reward = 0.0
                episode_steps = 0
                episode_intervention_steps = 0

                if n_target > 0 and episode_idx >= n_target:
                    patches._request_stop("EVAL", n_target)

                obs, info = env.reset()
                env_processor.reset()
                action_processor.reset()
                policy.reset()
                transition = env_processor(create_transition(observation=obs, info=info))

                if n_target > 0 and episode_idx >= n_target:
                    break
    except KeyboardInterrupt:
        print("[EVAL] stopping (Ctrl+C or auto-stop).", flush=True)
    finally:
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    from lerobot.configs import parser
    from lerobot.configs.train import TrainRLServerPipelineConfig

    from lerobot_control.hilserl.patches import apply_all_patches

    apply_all_patches()

    @parser.wrap()
    def _cli(cfg: TrainRLServerPipelineConfig):
        run_policy_eval(cfg)

    _cli()


if __name__ == "__main__":
    main()
