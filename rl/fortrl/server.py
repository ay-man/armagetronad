"""Launches a dedicated server with the RL bridge enabled and speaks its protocol.

See rl/PROTOCOL.md for the message format.
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1


class ServerError(RuntimeError):
    pass


class GameServer:
    """One dedicated server process plus the socket it connects back to.

    steered_teams: AI players on teams with id < steered_teams are controlled
    through the bridge; the rest keep the game's built-in AI.
    """

    def __init__(self, binary, game_config, steered_teams, port, extra_settings=None,
                 workdir=None, connect_timeout=60.0, keep_logs=False):
        self.binary = Path(binary).absolute()
        if not self.binary.exists():
            raise ServerError(
                f"dedicated server not found at {self.binary}; build it first (see rl/README.md)")
        self.game_config = Path(game_config)
        self.steered_teams = steered_teams
        self.port = port
        self.extra_settings = dict(extra_settings or {})
        self.connect_timeout = connect_timeout
        self.keep_logs = keep_logs
        self._own_workdir = workdir is None
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="fortrl_"))
        self.proc = None
        self.conn = None
        self._buffer = b""

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        cfgdir = self.workdir / "cfg"
        vardir = self.workdir / "var"
        cfgdir.mkdir(parents=True, exist_ok=True)
        vardir.mkdir(parents=True, exist_ok=True)
        sock_path = self.workdir / "bridge.sock"
        if sock_path.exists():
            sock_path.unlink()

        settings = {
            "RL_BRIDGE_SOCKET": str(sock_path),
            "RL_BRIDGE_TEAMS": self.steered_teams,
            "SERVER_PORT": self.port,
        }
        settings.update(self.extra_settings)
        lines = [self.game_config.read_text(), "", "# added by fortrl"]
        lines += [f"{k} {v}" for k, v in settings.items()]
        (cfgdir / "autoexec.cfg").write_text("\n".join(lines) + "\n")

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(sock_path))
        listener.listen(1)
        listener.settimeout(self.connect_timeout)

        self.log_path = self.workdir / "server.log"
        self._log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [str(self.binary), "--userconfigdir", str(cfgdir), "--vardir", str(vardir)],
            cwd=str(self.binary.parent), stdin=subprocess.DEVNULL,
            stdout=self._log_file, stderr=subprocess.STDOUT)
        log.info("started server pid %d port %d in %s", self.proc.pid, self.port, self.workdir)

        try:
            self.conn, _ = listener.accept()
        except socket.timeout:
            tail = self._log_tail()
            self.keep_logs = True
            self.close()
            raise ServerError(f"server did not connect within {self.connect_timeout}s; "
                              f"see {self.log_path}:\n{tail}")
        finally:
            listener.close()
        self.conn.settimeout(None)
        return self

    def close(self):
        if self.conn is not None:
            try:
                self.conn.sendall(b"QUIT\n")
            except OSError:
                pass
            self.conn.close()
            self.conn = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            self._log_file.close()
        if self._own_workdir and not self.keep_logs:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ protocol

    def fileno(self):
        return self.conn.fileno()

    def receive(self):
        """Blocks until the next message from the server arrives."""
        while b"\n" not in self._buffer:
            chunk = self.conn.recv(1 << 20)
            if not chunk:
                raise ServerError(f"server closed the connection; see {self.log_path}:\n"
                                  f"{self._log_tail()}")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        msg = json.loads(line)
        if msg.get("version") != PROTOCOL_VERSION:
            raise ServerError(f"protocol version {msg.get('version')} != {PROTOCOL_VERSION}; "
                              "rebuild the server")
        return msg

    def has_buffered(self):
        return b"\n" in self._buffer

    def send_actions(self, actions):
        """actions: {player_id: (turn, brake)} with turn in -1/0/1 and brake in 0/1."""
        parts = ["A"]
        for pid, (turn, brake) in actions.items():
            parts.append(f"{int(pid)} {int(turn)} {int(brake)}")
        self.conn.sendall((" ".join(parts) + "\n").encode())

    def send_ok(self):
        self.conn.sendall(b"OK\n")

    def _log_tail(self, lines=20):
        try:
            self._log_file.flush()
            text = Path(self.log_path).read_text(errors="replace").splitlines()
            return "\n".join(t for t in text[-lines:])
        except OSError:
            return "(no log)"


def default_binary():
    root = Path(__file__).resolve().parents[2]
    return os.environ.get("FORTRL_SERVER", str(root / "build" / "rl_server" / "armagetronad-dedicated"))

