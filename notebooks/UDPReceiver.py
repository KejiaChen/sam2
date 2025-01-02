import socket
from collections import deque


class UDPReceiver:
    def __init__(self, host, port, file_path):
        """
        Initialize the UDP Receiver with a specific host and port.
        """
        self.host = host
        self.port = port
        self.buffer_size = 4096  # Adjust as needed
        self.file_path = file_path  # Path to save the data
        self.intersection_data = deque(maxlen=10)  # To store only the last 10 entries
        self.sock = None

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            print(f"Listening on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to initialize or bind socket: {e}")
            self.sock = None  # Ensure the socket is None if initialization fails

    def receive_and_parse_data(self):
        """
        Receive data over UDP, parse it, and save it to a file.
        
        Returns:
            tuple: Parsed data (intersection1, intersection2, tcp) as lists of floats.
        """
        while True:
            try:
                # Receive data
                data, addr = self.sock.recvfrom(self.buffer_size)
                message = data.decode().strip()  # Decode and remove extra spaces
                
                # Remove brackets if present and split the values
                message = message.replace("[", "").replace("]", "")  # Clean up brackets
                parsed_data = list(map(float, message.split(",")))  # Parse floats

                # Ensure the length is exactly 9
                if len(parsed_data) != 9:
                    print(f"Unexpected data length: {len(parsed_data)}. Skipping.")
                    continue

                # Split the data into intersection1, intersection2, and tcp
                intersection1 = parsed_data[:3]  # First three values
                intersection2 = parsed_data[3:6]  # Next three values
                tcp = parsed_data[6:9]           # Last three values

                # Add the parsed data to recent_data
                self.intersection_data.append({
                    "intersection1": intersection1,
                    "intersection2": intersection2,
                    "tcp": tcp
                })

                # Save recent data to file
                self.save_to_file()

                print(f"Parsed Intersection1: {intersection1}")
                print(f"Parsed Intersection2: {intersection2}")
                print(f"Parsed TCP: {tcp}")

                return intersection1, intersection2, tcp

            except ValueError as e:
                print(f"Error parsing data: {e}. Skipping.")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}. Exiting receiver loop.")
                break


    def save_to_file(self):
        """
        Save the last 10 entries to the file.
        """
        try:
            with open(self.file_path, "w") as file:
                for entry in self.intersection_data:
                    intersection1 = ",".join(map(str, entry["intersection1"]))
                    intersection2 = ",".join(map(str, entry["intersection2"]))
                    tcp = ",".join(map(str, entry["tcp"]))
                    file.write(f"{intersection1},{intersection2},{tcp}\n")
            print(f"Saved recent data to {self.file_path}")
        except Exception as e:
            print(f"Failed to save data to file: {e}")

    def read_latest_data(self):
        """
        Read the latest entry from the file.
        
        Returns:
            dict: The most recent entry as a dictionary.
        """
        try:
            with open(self.file_path, "r") as file:
                lines = file.readlines()
                if lines:
                    latest_line = lines[-1].strip()
                    parsed_data = list(map(float, latest_line.split(",")))
                    return {
                        "intersection1": parsed_data[:3],
                        "intersection2": parsed_data[3:6],
                        "tcp": parsed_data[6:9]
                    }
        except FileNotFoundError:
            print("Data file not found. No data to read.")
        except Exception as e:
            print(f"Failed to read data from file: {e}")

        return None

    def close_socket(self):
        """
        Close the socket if it's open.
        """
        if self.sock:
            print(f"Closing socket bound to {self.host}:{self.port}")
            self.sock.close()
            self.sock = None
            print("Socket closed successfully.")


# # Example Usage
if __name__ == "__main__":
    # Set up the receiver on the specified host and port
    udp_host = "10.157.175.179"  # Receiver's IP address
    udp_port = 5060            # Same port as used by the sender
    file_path = "/home/tp2/Documents/sam2/intersections.txt"  # Path to save the last 10 entries

    # Initialize the UDP receiver
    receiver = UDPReceiver(udp_host, udp_port, file_path)

    try:
        while True:
            # Receive and process data
            intersection1, intersection2, tcp = receiver.receive_and_parse_data()

            # if intersection1 and intersection2 and tcp:
            #     print(f"Processed Data:\nIntersection 1: {intersection1}\nIntersection 2: {intersection2}\nTCP: {tcp}")
            
            # # Optionally, read the latest data from the file
            # latest_data = receiver.read_latest_data()
            # if latest_data:
            #     print(f"Latest Data from File: {latest_data}")

    except KeyboardInterrupt:
        print("Receiver interrupted by user.")
    finally:
        # Ensure the socket is closed on exit
        receiver.close_socket()
