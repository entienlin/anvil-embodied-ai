"""save_actor_trajectory(): one .npz per episode with state/image/action/
reward/done/is_intervention, never raising into the control loop. Pure logic.

Run: python3 -m lerobot_control.hilserl._stage7g_actor_trajectory_saved_test
"""

import os
import sys
import tempfile

import numpy as np
import torch

from lerobot_control.hilserl.patches import save_actor_trajectory


def _transitions(n=5, with_image=True, with_flag=True):
    out = []
    for i in range(n):
        st = {"observation.state": torch.arange(7.0) + i}
        if with_image:
            st["observation.images.wrist_r"] = torch.full((1, 3, 8, 8), 0.1 * i)  # batched like the real pipeline
        t = {
            "state": st,
            "action": torch.tensor([0.1 * i, -0.2, 0.3]),
            "reward": torch.tensor(1.0 if i == n - 1 else 0.0),
            "done": torch.tensor(i == n - 1),
        }
        if with_flag:
            t["complementary_info"] = {"discrete_penalty": torch.tensor([1.0 if i >= 3 else 0.0])}
        out.append(t)
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        os.environ["HILSERL_TRAJ_DIR"] = os.path.join(d, "not_yet_created")
        path = save_actor_trajectory(_transitions())
        if path is None or not os.path.exists(path):
            print("[FAIL] no file written", flush=True)
            return 1
        z = np.load(path)
        if z["state_deg"].shape != (5, 7) or z["action"].shape != (5, 3) or z["reward"].sum() != 1.0 or z["done"][-1] != 1.0:
            print(f"[FAIL] wrong contents: { {k: z[k].shape for k in z.files} }", flush=True)
            return 1
        if z["image"].shape != (5, 3, 8, 8) or z["image"].dtype != np.float16 or not np.isclose(float(z["image"][2].mean()), 0.2, atol=1e-3):
            print(f"[FAIL] image wrong: {z['image'].shape} {z['image'].dtype}", flush=True)
            return 1
        if list(z["is_intervention"]) != [0, 0, 0, 1, 1]:
            print(f"[FAIL] intervention flag wrong: {z['is_intervention']}", flush=True)
            return 1
        print("[OK] episode saved with state/image/action/reward/done/is_intervention; batched image squeezed", flush=True)

        p2 = save_actor_trajectory(_transitions(with_flag=False, with_image=False))
        z2 = np.load(p2)
        if "image" in z2.files or z2["is_intervention"].sum() != 0:
            print("[FAIL] missing image/flag should degrade gracefully", flush=True)
            return 1
        print("[OK] missing image / flag degrades gracefully", flush=True)

        os.environ["HILSERL_TRAJ_DIR"] = "/proc/not/writable"
        if save_actor_trajectory(_transitions()) is not None or save_actor_trajectory([{"bad": 1}]) is not None:
            print("[FAIL] failures must be swallowed and return None", flush=True)
            return 1
        print("[OK] failures are swallowed (returns None, control loop unaffected)", flush=True)
    print("[stage7g] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
