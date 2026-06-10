import socket
import threading
import json
from collections import deque
import time


class GameStateReceiver:
    """Receives gamestate from JumpKing C# mod via TCP socket.
    
    Runs a background thread that continuously reads incoming gamestate
    messages and stores the latest one. Python reads from the buffer
    whenever it needs current gamestate — no file polling needed.
    
    Also handles sending teleport commands back to C#.
    """
    _instance = None

    def __init__(self, host="127.0.0.1", port=7777, max_retries=30):
        print(f"GameStateReceiver created, connecting to {host}:{port}")
        self.host = host
        self.port = port
        self.max_retries = max_retries

        # latest gamestate — deque(maxlen=1) always has most recent
        self._buffer = deque(maxlen=1)
        self._lock = threading.Lock()

        self._socket = None
        self._running = False
        self._receive_thread = None
        self._partial = ""  # handle partial messages

        self.connect()

    @classmethod
    def get_shared(cls, host="127.0.0.1", port=7777):
        if cls._instance is None:
            cls._instance = cls(host=host, port=port)
        return cls._instance

    @classmethod
    def reset_shared(cls):
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None

    def connect(self):
        """Connect to C# TCP server with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._running = True
                self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self._receive_thread.start()
                print(f"Connected to game on port {self.port}")
                return
            except ConnectionRefusedError:
                if attempt < self.max_retries - 1:
                    print(f"Waiting for game... ({attempt + 1}/{self.max_retries})")
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Could not connect to game after {self.max_retries} attempts")

    def _receive_loop(self):
        """Background thread — continuously reads gamestate messages."""
        while self._running:
            try:
                data = self._socket.recv(4096).decode('utf-8')
                if not data:
                    raise ConnectionError("Server closed connection")

                # handle multiple messages in one recv or partial messages
                self._partial += data
                while '\n' in self._partial:
                    line, self._partial = self._partial.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        with self._lock:
                            self._buffer.append(parsed)
                    except json.JSONDecodeError:
                        pass

            except Exception:
                if self._running:
                    print("Connection lost, reconnecting...")
                    time.sleep(1)
                    try:
                        self._partial = ""  # clear partial buffer on reconnect
                        self.connect()
                    except:
                        pass
                break

    def read_gamedata(self):
        """Returns the latest gamestate dict, or None if none received yet."""
        with self._lock:
            if self._buffer:
                return self._buffer[0]
        return None

    def wait_for_landing(self, jumped, prev_write_count, wait_for_v0, timeout=15.0):
        if not jumped:
            time.sleep(0.05)
            # check if character fell off edge during walk
            data = self.read_gamedata()
            if data is not None and not data.get("is_on_ground"):
                # character fell — wait for genuine landing
                jumped = True
            else:
                return
        
        start = time.time()
        # phase 1: wait for character to leave ground
        if jumped:
            while time.time() - start < timeout:
                data = self.read_gamedata()
                if data is not None and not data.get("is_on_ground"):
                    break
                time.sleep(0.005)
        
        # phase 2: wait for genuine landing via write_count
        while time.time() - start < timeout:
            data = self.read_gamedata()
            if data is not None and data.get("is_on_ground") and \
            data.get("write_count", 0) > prev_write_count:
                
                # if short hop after large jump, wait for velocity to reach 0
                if wait_for_v0:
                    v0_start = time.time()
                    while time.time() - v0_start < 3.0:
                        v_data = self.read_gamedata()
                        if v_data is not None:
                            vel_x = abs(float(v_data.get("vel_x", 0)))
                            if vel_x < 0.01:
                                break
                        time.sleep(0.005)
                    wait_for_v0 = False
                return
            time.sleep(0.005)
        
        print(f"wait_for_landing timed out after {timeout}s")

    def send_teleport(self, x, y):
        """Sends a teleport command to C#."""
        if self._socket is None or not self._running:
            return
        try:
            cmd = f"teleport:{x:.2f},{y:.2f}\n"
            self._socket.sendall(cmd.encode('utf-8'))
        except Exception as e:
            print(f"Teleport send error: {e}")

    def close(self):
        """Closes the connection."""
        self._running = False
        try:
            self._socket.close()
        except Exception:
            pass

if __name__ == "__main__":
    # quick test
    receiver = GameStateReceiver()
    print("Waiting for gamestate...")
    for _ in range(10):
        data = receiver.read_gamedata()
        if data:
            print(f"x={data['x']}, y={data['y']}, screen={data['current_screen']}, "
                  f"on_ground={data['is_on_ground']}, write_count={data['write_count']}")
        time.sleep(0.1)
    receiver.close()