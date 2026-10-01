"""Toggle-based keyboard listener for HIL-SERL intervention control.

Deliberately NOT reusing lerobot.teleoperators.keyboard's stock listener:
that one treats "any movement key currently held down" as the intervention
signal (see teleop_keyboard.py's get_teleop_events()), which is the wrong
model here -- the leader arm itself provides the actual intervention
motion, so the keyboard's only job is a discrete press-once-to-take-over /
press-again-to-give-back toggle, plus simple success/failure/rerecord
labels for the "no reward classifier yet" v1 (see the plan's Part 2.6).

Reads raw keypresses directly off stdin (termios/tty cbreak mode), NOT
pynput -- this class was originally pynput-based, which needs a real X11
DISPLAY (confirmed the hard way: crashed on the GPU PC, a headless server
with no X server at all, during the first real Stage 5 attempt). This
machine is accessed over SSH/terminal, not a GUI desktop, so a real TTY
(docker run -it gives the container one, connected to the caller's actual
terminal) is the right input source, not X11. Stock lerobot's own
keyboard_ee teleoperator (used only for Stage 3's one-off dry run, never
for real hardware) is still pynput-based and still needs xvfb for that --
unrelated to this class, not changed.
"""

import atexit
import select
import sys
import termios
import threading
import tty


class KeyboardIntervention:
    """Tracks intervention-toggle and episode-label key state.

    Keys:
      space -- toggle intervention on/off (press once to take over, press
               again to give back to the policy)
      s     -- mark success
      f     -- mark failure
      r     -- mark rerecord
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._is_intervention = False
        self._success = False
        self._failure = False
        self._rerecord = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._orig_term_settings = None

    def start(self) -> None:
        # Idempotent: a second start() while a reader thread is already
        # running is a no-op rather than spawning a second thread. Defense
        # in depth for bimanual/multi-arm use (see
        # teleop_anvil_openarm_multi_leader.py) -- the composite leader is
        # responsible for calling start()/stop() exactly once on a shared
        # instance, but a stray extra call here must not reintroduce the
        # exact stdin-race this class's module docstring warns about (two
        # threads both doing blocking sys.stdin.read(1) on the same fd).
        if self._thread is not None and self._thread.is_alive():
            return
        if not sys.stdin.isatty():
            raise RuntimeError(
                "KeyboardIntervention needs a real interactive TTY -- run the "
                "container with `docker run -it` (or `-t -i`), not detached or "
                "with stdin piped/redirected."
            )

        self._orig_term_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        # Safety net matching this package's established atexit pattern
        # (see robot_anvil_openarm.py/teleop_anvil_openarm_leader.py's
        # connect()): if the interpreter exits without calling stop(), the
        # terminal must not be left in cbreak mode for whatever shell the
        # user returns to.
        atexit.register(self._restore_terminal)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch == " ":
                with self._lock:
                    self._is_intervention = not self._is_intervention
            elif ch == "s":
                with self._lock:
                    self._success = True
            elif ch == "f":
                with self._lock:
                    self._failure = True
            elif ch == "r":
                with self._lock:
                    self._rerecord = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self._orig_term_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_term_settings)
            self._orig_term_settings = None

    @property
    def is_intervention(self) -> bool:
        with self._lock:
            return self._is_intervention

    def reset_intervention(self) -> None:
        """Clear the space-bar toggle back to "not intervening". The toggle
        is otherwise persistent across episodes (s/f never touch it), so
        without this a human who intervened in one episode would silently
        start the next one already in control.

        Also drops any s/f/r pressed during the homing window: RobotEnv.reset()
        never reads them, so they would otherwise latch and label the FIRST
        step of the next episode (seen 2026-09-21: 1-step "success" episode)."""
        with self._lock:
            self._is_intervention = False
            self._success = False
            self._failure = False
            self._rerecord = False

    def consume_episode_labels(self) -> dict:
        """Read and clear the one-shot success/failure/rerecord flags."""
        with self._lock:
            labels = {
                "success": self._success,
                "failure": self._failure,
                "rerecord": self._rerecord,
            }
            self._success = False
            self._failure = False
            self._rerecord = False
            return labels
