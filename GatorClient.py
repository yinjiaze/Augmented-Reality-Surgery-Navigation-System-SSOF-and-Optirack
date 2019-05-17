import socket
from threading import Thread



class GatorClient:

    def __init__(self, dataPort= 61556, SensorListener= None, steaming_memory=None):

        self.hostaddress = socket.gethostbyname(socket.gethostname())
        self.dataPort = dataPort
        self.SensorListener = SensorListener
        self.steaming_memory = steaming_memory
        self.running = False

    # Create a data socket to attach to the SSOF stream

    def __createDataSocket(self, hostaddress, port):

        result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP

        result.bind((hostaddress, port))

        return result

    # Unpack Sensor Data

    def __unpackSensorData(self, data, steaming_memory):

        datalist = data.split()
        #pos2x, pos2y, pos3z = datalist[20:23]
        pos2x, pos2y, pos3z = datalist[2:5]
        if self.SensorListener is not None:
            self.SensorListener(pos2x, pos2y, pos3z, steaming_memory)

    def __dataThreadFunction(self, sock):

        sock.settimeout(0.01)

        while self.running:

            # Block for input

            try:

                data, addr = sock.recvfrom(32768)  # 32k byte buffer size
                data = data.decode('utf-8')
                if (len(data) >= 26):
                    self.__processMessage(data)

            except socket.timeout:

                pass

    def __processMessage(self, data):

        self.__unpackSensorData(data, self.steaming_memory)


    def run(self):

        # Set running flag to True

        self.running = True

        # Create the data socket

        self.dataSocket = self.__createDataSocket(self.hostaddress, self.dataPort)

        if (self.dataSocket is None):
            print("Could not open data channel")

            exit

        dataThread = Thread(target=self.__dataThreadFunction, args=(self.dataSocket,))

        dataThread.start()


    def stop(self):

        self.running = False

