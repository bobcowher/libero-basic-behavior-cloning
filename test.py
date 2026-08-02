"""Watch the policy run in a live viewer window.

    python test.py                 # newest checkpoint, scene 0
    python test.py --scene 3       # a different init state
    python test.py --ckpt checkpoints/run428_ablation_b256

For a success rate use eval.py; for MP4s of a batch of rollouts,
eval.py's Agent.evaluate(record_dir=...).
"""
import os

# Must precede the agent import: mujoco picks its rendering backend when it is
# imported, and the conda env sets MUJOCO_GL=egl for headless work, which
# cannot open a window.
os.environ["MUJOCO_GL"] = "glfw"
# PYOPENGL_PLATFORM must be unset or "egl" -- robosuite raises ImportError on
# anything else (renderers/context/egl_context.py). The conda env sets it to
# egl, so clear it rather than point it at glfw.
os.environ.pop("PYOPENGL_PLATFORM", None)

import argparse  # noqa: E402

from agent import Agent  # noqa: E402
from eval import latest_ckpt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="default: newest file in checkpoints/")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--scene", type=int, default=None,
                   help="benchmark init state to start from; omit to use the "
                        "reference project's seed-0 double-reset scene")
    p.add_argument("--max-steps", type=int, default=900)
    args = p.parse_args()

    ckpt = args.ckpt or latest_ckpt()

    agent = Agent(task_id=args.task, ckpt=ckpt)
    print(f"task {args.task}: {agent.task.language}")
    print(f"checkpoint: {ckpt}")

    agent.load_checkpoint()
    agent.test(scene=args.scene, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
