"""Watch the policy run, in a live window, on the scenes eval.py scores.

    python test.py                      # newest checkpoint, scenes 0,10,20,30,40
    python test.py --scenes 0-9         # a range
    python test.py --scenes 6,23,41     # specific scenes
    python test.py --ckpt checkpoints/run428_ablation_b256

This is a VIEW of the eval, not a separate measurement: every rollout runs
through Agent.evaluate, so a scene you see succeed is a scene eval.py counts.
The printed tally is a liveness read -- for a number, use eval.py --n-eval 50.

Verify the two agree with:

    python test.py --scenes 0,10,20,30,40
    python eval.py --scenes 0,10,20,30,40
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
from eval import latest_ckpt, parse_scenes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="default: newest file in checkpoints/")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--scenes", default=None,
                   help='init states to watch, e.g. "0-9" or "0,10,20"; '
                        "default is a stride across all 50")
    p.add_argument("--max-steps", type=int, default=300,
                   help="successes land near 100 steps; the cap is how long "
                        "you wait on a failure")
    p.add_argument("--reference-protocol", action="store_true",
                   help="reproduce the reference project's double-reset scene; "
                        "not one of the 50 scored scenes")
    args = p.parse_args()

    ckpt = args.ckpt or latest_ckpt()

    agent = Agent(task_id=args.task, ckpt=ckpt)
    print(f"task {args.task}: {agent.task.language}")
    print(f"checkpoint: {ckpt}")

    agent.load_checkpoint()
    agent.test(scenes=parse_scenes(args.scenes) if args.scenes else None,
               max_steps=args.max_steps,
               reference_protocol=args.reference_protocol)


if __name__ == "__main__":
    main()
