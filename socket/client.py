#!/usr/bin/env python3

import time
import socket
import logging

from numpy import byte

import cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class RawDataSocketClient:
    """A simple socket client that sends/receives raw data to/from server"""

    def __init__(self, name: str, server_ip_addr: str = "127.0.0.1", server_port: int = cfg.PORT) -> None:
        self.name = name
        self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info("[%s] Connecting %s:%s", self.name,
                    server_ip_addr, server_port)
        self.skt.connect((server_ip_addr, server_port))

    def send(self, data: bytearray, packet_size: int = 0, wait_ms: int = 0) -> None:
        logger.debug("[%s] > send data_size=%d packet_size=%d wait_ms=%d",
                     self.name, len(data), packet_size, wait_ms)
        data_size = len(data)
        sent_size = 0
        while sent_size < data_size:
            if (packet_size == 0) or (packet_size > data_size):
                packet = data
            else:
                packet = data[sent_size:sent_size+packet_size]
            self.skt.sendall(packet, socket.MSG_WAITALL)
            logger.debug("[%s] - Sent packet size = %d",
                         self.name, len(packet))

            sent_size += len(packet)
            if wait_ms:
                time.sleep(wait_ms * 0.001)


def main():
    """Main"""
    clt = RawDataSocketClient("CLIENT")
    data = bytearray(
        "The freedom of Speech may be taken away, and, dumb and silent we may be led, like sheep, to the Slaughter.", encoding="utf-8")
    clt.send(data)
    print(data)


if __name__ == "__main__":
    main()
