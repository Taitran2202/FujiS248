import threading
import zmq

class ZMQSocketListener:
    __instance = None

    def __init__(self, ip_address, port):
        if ZMQSocketListener.__instance is not None:
            raise Exception("ZMQSocketListener is a singleton!")
        else:
            ZMQSocketListener.__instance = self
            self.ip_address = ip_address
            self.port = port
            self.shutdown = threading.Event()
            # self.shutdown.clear()

    @classmethod
    def get_instance(cls, ip_address, port):
        if cls.__instance is None:
            cls(ip_address, port)
        return cls.__instance
    
    def listen(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.ip_address}:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.shutdown.clear()
        while True:
            if self.shutdown.is_set():
                break
            socks = dict(self.poller.poll(1000))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                message_topic = self.socket.recv()
                message = self.socket.recv()
                yield message_topic,message
        self.dispose()
        yield b'exit',b''

    def dispose(self):
        self.poller.unregister(self.socket)
        self.socket.close()
        self.context.term()
