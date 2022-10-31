#!/usr/bin/env python3

import time
import socket
import logging

import cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def get_local_ip_addr() -> str:
    _, _, ls_addr = socket.gethostbyname_ex(socket.gethostname())
    for addr in ls_addr:
        if addr.startswith("192.168."):
            return addr
    raise RuntimeError(
        "Cannot find local IP address starts with 192.168, availables are %s", ls_addr)


class SingleClientRawDataSocketServer:
    """A simple socket server that receives/sends raw data from/to one single client"""

    def __init__(self, name: str, server_ip_addr: str = "127.0.0.1", server_port: int = cfg.PORT) -> None:
        self.name = name

        self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if len(server_ip_addr) == 0:
            server_ip_addr = get_local_ip_addr()
        self.skt.bind((server_ip_addr, server_port))
        self.skt.listen()
        logger.info("[%s] Listening %s:%s", self.name,
                    server_ip_addr, server_port)

        self.conn, addr = self.skt.accept()
        logger.info("[%s] Connection from %s", self.name, addr)

    def recv(self, size: int, wait_ms: int = 0) -> bytearray:
        logger.debug("[%s] > recv size=%d wait_ms=%d",
                     self.name, size, wait_ms)
        data = bytearray()
        while len(data) < size:
            packet = self.conn.recv(size)
            if not packet:
                logger.warning("[%s] - No packet", self.name)
                break
            else:
                logger.debug("[%s] - Received packet size = %d",
                             self.name, len(packet))
                data.extend(packet)
            if wait_ms:
                time.sleep(wait_ms * 0.001)
        return data


def main():
    """Main"""
    svr = SingleClientRawDataSocketServer("SERVER")
    data = svr.recv(cfg.RAW_PACKET_SIZE)
    print(data)


if __name__ == "__main__":
    main()
