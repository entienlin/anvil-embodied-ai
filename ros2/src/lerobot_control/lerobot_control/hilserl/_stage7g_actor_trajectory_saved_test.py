"""save_actor_trajectory(): one .npz per episode with state/action/reward/done,
never raising into the control loop. Pure logic.

Run: python3 -m lerobot_control.hilserl._stage7g_actor_trajectory_saved_test
"""

import os
import sys
import tempfile

import numpy as np
import torch

from lerobot_control.hilserl.patches import save_actor_trajectory


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        os.environ["HILSERL_TRAJ_DIR"] = os.path.join(d, "not_yet_created")
        transitions = [
            {
                "state": {"observation.state": torch.arange(7.0) + i, "observation.images.wrist_r": torch.zeros(3, 4, 4)},
                "action": torch.tensor([0.1 * i, -0.2, 0.3]),
                "reward": torch.tensor(1.0 if i == 4 else 0.0),
                "done": torch.tensor(i == 4),
            }
            for i in range(5)
        ]
        path = save_actor_trajectory(transitions)
        if path is None or not os.path.exists(path):
            print("[FAIL] no file written", flush=True)
            return 1
        z = np.load(path)
        if z["state_deg"].shape != (5, 7) or z["action"].shape != (5, 3) or z["reward"].sum() != 1.0 or z["done"][-1] != 1.0:
            print(f"[FAIL] wrong contents: { {k: z[k].shape for k in z.files} }", flush=True)
            return 1
        if not np.allclose(z["state_deg"][2], np.arange(7.0) + 2) or not np.isclose(z["action"][3][0], 0.3):
            print("[FAIL] values not preserved", flush=True)
            return 1
        print("[OK] episode saved with state/action/reward/done, directory auto-created", flush=True)

        os.environ["HILSERL_TRAJ_DIR"] = "/proc/not/writable"
        if save_actor_trajectory(transitions) is not None:
            print("[FAIL] should return None when the directory can't be written", flush=True)
            return 1
        if save_actor_trajectory([{"bad": 1}]) is not None:
            print("[FAIL] malformed transitions must be swallowed", flush=True)
            return 1
        print("[OK] failures are swallowed (returns None, control loop unaffected)", flush=True)
    print("[stage7g] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
