# Echo server program
import socket
import random

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 5204              # Arbitrary non-privileged port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with conn:

        print('Connected by', addr)
        data_old = ''
        while True:
            #data = conn.recv(1024) #1024

            with open("valence.json",'r') as valence_file:
                data = valence_file.read()
                #data = str(random.randint(0,500))+'/'+str(random.randint(0,500))+'/'

            #data = bytes(random.randint(0,10))
            #if not data: break
                if data != data_old:

                    print(data)
                    data_encode = data.encode()

                    conn.sendall(data_encode)
                    data_old = data
