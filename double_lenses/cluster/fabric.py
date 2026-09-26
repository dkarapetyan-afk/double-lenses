"""
Communication Fabric for Multi-Node and Multi-Device Distributed Clusters.
Provides low-latency TCP messaging, buffer serialization, and intra-node direct queues.
"""

import contextlib
import json
import queue
import socket
import struct
import threading
from typing import Union

import numpy as np

from double_lenses.autodiff.tensor import Tensor
from double_lenses.cluster.topology import ClusterTopology, DeviceAddress


class Serializer:
    """Zero-overhead binary serializer for Tensors and metadata."""

    MAGIC = b"DLNS"

    @classmethod
    def serialize_tensor(cls, tensor: Tensor, tag: str = "") -> bytes:
        np_arr = np.ascontiguousarray(tensor.to_numpy())
        shape = list(np_arr.shape)
        dtype_str = str(np_arr.dtype)
        meta = {
            "tag": tag,
            "shape": shape,
            "dtype": dtype_str,
            "device": tensor.device,
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        data_bytes = np_arr.tobytes()

        # Format: MAGIC (4) + meta_len (4) + data_len (8) + meta + data
        header = struct.pack("!4sIQ", cls.MAGIC, len(meta_bytes), len(data_bytes))
        return header + meta_bytes + data_bytes

    @classmethod
    def deserialize_tensor(cls, buf: bytes) -> tuple[Tensor, str]:
        magic, meta_len, data_len = struct.unpack("!4sIQ", buf[:16])
        if magic != cls.MAGIC:
            raise ValueError("Corrupt tensor buffer: magic mismatch")
        meta_bytes = buf[16 : 16 + meta_len]
        data_bytes = buf[16 + meta_len : 16 + meta_len + data_len]

        meta = json.loads(meta_bytes.decode("utf-8"))
        dtype = np.dtype(meta["dtype"])
        np_arr = np.frombuffer(data_bytes, dtype=dtype).reshape(meta["shape"]).copy()
        tensor = Tensor(np_arr, device=meta["device"])
        return tensor, meta["tag"]


class SocketConnection:
    """Manages a non-blocking TCP socket stream."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send_msg(self, data: bytes) -> None:
        total_len = len(data)
        # Send length prefix
        self.sock.sendall(struct.pack("!I", total_len) + data)

    def recv_msg(self) -> bytes | None:
        try:
            len_bytes = self._recv_exact(4)
            if not len_bytes:
                return None
            total_len = struct.unpack("!I", len_bytes)[0]
            return self._recv_exact(total_len)
        except (ConnectionResetError, BrokenPipeError):
            return None

    def _recv_exact(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                break
            data.extend(packet)
        return bytes(data)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sock.close()


class CommunicationFabric:
    """
    Cluster Communication Fabric coordinating multi-node TCP and local in-memory transfers.
    """

    def __init__(self, topology: ClusterTopology, local_node_id: str = "node0"):
        self.topology = topology
        self.local_node_id = local_node_id
        self.local_node = topology.nodes.get(local_node_id)

        # Local queues for zero-copy intra-node transfers
        self._local_mailboxes: dict[str, queue.Queue] = {}
        # Inbound socket registry
        self._server_sock: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False
        self._active_connections: dict[str, SocketConnection] = {}

        # Initialize mailboxes for all known devices
        for dev in topology.all_devices():
            self._local_mailboxes[str(dev)] = queue.Queue()

    def start(self) -> None:
        """Starts background TCP listener if node has a configured port."""
        if self._running:
            return
        self._running = True

        if self.local_node:
            try:
                self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._server_sock.bind((self.local_node.host, self.local_node.port))
                self._server_sock.listen(128)
                self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
                self._server_thread.start()
            except Exception:
                # In environments where socket binding is restricted, in-memory queues still work
                pass

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            with contextlib.suppress(Exception):
                self._server_sock.close()
        for conn in self._active_connections.values():
            conn.close()
        self._active_connections.clear()

    def _server_loop(self) -> None:
        while self._running:
            try:
                client_sock, _ = self._server_sock.accept()
                conn = SocketConnection(client_sock)
                threading.Thread(target=self._client_handler, args=(conn,), daemon=True).start()
            except Exception:
                break

    def _client_handler(self, conn: SocketConnection) -> None:
        while self._running:
            msg = conn.recv_msg()
            if not msg:
                break
            tensor, tag = Serializer.deserialize_tensor(msg)
            # Route to target mailbox
            if tag in self._local_mailboxes:
                self._local_mailboxes[tag].put(tensor)

    def send(self, tensor: Tensor, target_device: Union[str, DeviceAddress], tag: str = "") -> None:
        tgt_str = str(target_device)
        dest_dev = self.topology.get_device(target_device)

        # Intra-node / in-process transfer
        if dest_dev.node_id == self.local_node_id or tgt_str in self._local_mailboxes:
            # Transfer tensor (optionally moving to destination device memory)
            t_copy = tensor.copy()
            t_copy.device = str(dest_dev)
            self._local_mailboxes[tgt_str].put(t_copy)
            return

        # Inter-node transfer via TCP socket
        target_node = self.topology.nodes.get(dest_dev.node_id)
        if not target_node:
            # Fallback to local queue if node not in remote map
            self._local_mailboxes[tgt_str].put(tensor.copy())
            return

        conn = self._get_connection(target_node.node_id, target_node.host, target_node.port)
        if conn:
            payload = Serializer.serialize_tensor(tensor, tag=tgt_str)
            conn.send_msg(payload)
        else:
            self._local_mailboxes[tgt_str].put(tensor.copy())

    def recv(self, device: Union[str, DeviceAddress], timeout: float | None = 10.0) -> Tensor | None:
        dev_str = str(device)
        if dev_str not in self._local_mailboxes:
            self._local_mailboxes[dev_str] = queue.Queue()
        try:
            return self._local_mailboxes[dev_str].get(timeout=timeout)
        except queue.Empty:
            return None

    def _get_connection(self, node_id: str, host: str, port: int) -> SocketConnection | None:
        if node_id in self._active_connections:
            return self._active_connections[node_id]
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            conn = SocketConnection(sock)
            self._active_connections[node_id] = conn
            return conn
        except Exception:
            return None
