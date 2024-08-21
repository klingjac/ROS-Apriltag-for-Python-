import socket
import numpy as np
import cv2
from picamera2 import Picamera2
import ctypes
import os
import time

class Pose(ctypes.Structure):
    _fields_ = [
        ("tvec", ctypes.c_double * 3),
        ("rod_vect", ctypes.c_double * 3)
    ]

class Detection(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("pose", Pose),
        ("valid", ctypes.c_bool)
    ]

class Detections(ctypes.Structure):
    _fields_ = [
        ("detection_vect", Detection * 6)
    ]

# Load the shared library
lib = ctypes.CDLL(os.path.abspath("libpose_estimation.so"))

# Define the function prototype
lib.estimate_pose_and_draw.restype = Detections
lib.estimate_pose_and_draw.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]

def combine_rodrigues_vectors(rod_vect1, rod_vect2):
    """
    Combine two rotations given by Rodrigues vectors.
    
    Parameters:
    rod_vect1 (list or numpy array): The first Rodrigues vector.
    rod_vect2 (list or numpy array): The second Rodrigues vector.
    
    Returns:
    numpy array: The Rodrigues vector representing the combined rotation.
    """
    # Ensure the inputs are numpy arrays of type float64
    rod_vect1 = np.array(rod_vect1, dtype=np.float64)
    rod_vect2 = np.array(rod_vect2, dtype=np.float64)
    
    # Convert the first Rodrigues vector to a rotation matrix
    rotation_matrix1, _ = cv2.Rodrigues(rod_vect1)
    
    # Convert the second Rodrigues vector to a rotation matrix
    rotation_matrix2, _ = cv2.Rodrigues(rod_vect2)
    
    # Combine the rotations by matrix multiplication
    combined_rotation_matrix = np.dot(rotation_matrix2, rotation_matrix1)
    
    # Convert the combined rotation matrix back to a Rodrigues vector
    combined_rod_vect, _ = cv2.Rodrigues(combined_rotation_matrix)
    
    return combined_rod_vect.flatten()

def rodrigues_to_matrix(rod_vect):
    """
    Convert a Rodrigues vector to a rotation matrix.
    
    Parameters:
    rod_vect (list or numpy array): The Rodrigues vector.
    
    Returns:
    numpy array: The rotation matrix.
    """
    rod_vect = np.array(rod_vect, dtype=np.float64)
    rotation_matrix, _ = cv2.Rodrigues(rod_vect)
    return rotation_matrix

def matrix_to_euler(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles.
    
    Parameters:
    rotation_matrix (numpy array): The rotation matrix.
    
    Returns:
    tuple: The Euler angles (yaw, pitch, roll) in radians.
    """
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return yaw, pitch, roll

def rodrigues_to_euler(rod_vect):
    """
    Convert a Rodrigues vector to Euler angles.
    
    Parameters:
    rod_vect (list or numpy array): The Rodrigues vector.
    
    Returns:
    tuple: The Euler angles (yaw, pitch, roll) in radians.
    """
    rotation_matrix = rodrigues_to_matrix(rod_vect)
    euler_angles = matrix_to_euler(rotation_matrix)
    return euler_angles





def main():
    # Camera intrinsic parameters (from calibration)
    fx = 851.07105717
    fy = 844.9721067
    cx = 250.68622123
    cy = 227.71466514
    tag_size = 0.056  # Size of the AprilTag in meters

    # Initialize Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))
    picam2.start()

    # Initialize UDP socket
    udp_ip = "192.168.0.122"  # Replace with the IP address of the ROS2 node or listener
    udp_port = 5005  # Replace with the port number of the ROS2 node or listener
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -------------------- Base rotation ---------------------------
    # Default conventions are that the "Measured" Z-axis points out of the camers leaving the Y-axis pointing Down
    # This requires a transform of coordinates to get us into the "vehicle frame"
    # Achieving this can be done by making two 90 degree rotations abou the X & Y axises respectively



    # Initialize tag pose dictionary - Rotation - Position (based on tag center)
    poses = {
        4: [combine_rodrigues_vectors([np.pi/2, 0, 0], [0, 0, 0]), [0, 0, 0]],  # Z+ face (no rotation)
        1: [combine_rodrigues_vectors([0, 0, 0], [0, np.pi/2, 0]), [0, 0, 0]],  # Y+ face (90 degrees around X-axis)
        0: [combine_rodrigues_vectors([0, 0, 0], [0, 0, 0]), [0, 0, 0]],  # X+ face (90 degrees around Y-axis)
        3: [combine_rodrigues_vectors([0, 0, 0], [0, -np.pi/2, 0]), [0, 0, 0]],  # Y- face (-90 degrees around X-axis)
        2: [combine_rodrigues_vectors([0, 0, 0], [0, np.pi, 0]), [0, 0, 0]],  # X- face (-90 degrees around Y-axis)
        5: [combine_rodrigues_vectors([0, 0, 0], [np.pi, 0, 0]), [0, 0, 0]]  # Z- face (180 degrees around X-axis)
    }

    ##################################################################################################################
    #
    #   Mapping Axis Table: (Axis is out from the Numbered Face)
    #   0: X
    #   1: Y
    #   2: -X
    #   3: -Y
    #   4: Z
    #   5: -Z
    #
    ##################################################################################################################

    while True:
        iteration_beginning = time.time()
        frame = picam2.capture_array()
        height, width, channels = frame.shape
        frame_contig = np.ascontiguousarray(frame)
        frame_ptr = frame_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        detection_time = time.time()
        
        detections = lib.estimate_pose_and_draw(frame_ptr, width, height, fx, fy, cx, cy, tag_size)
        
        post_detect_time = time.time()
        
        sent_udp = False
        
        for detection in detections.detection_vect:
            if detection.valid and not sent_udp:
                # Extract the Rodrigues vector of the first valid detection
                rod_vect = [detection.pose.rod_vect[0], detection.pose.rod_vect[1], detection.pose.rod_vect[2]]

                # Continue with the usual processing and display
                tag_id = detection.id

                combined_rotation = combine_rodrigues_vectors(poses[tag_id][0], rod_vect)
                # Convert the Rodrigues vector to a string for transmission
                if tag_id == 0:
                    rod_vect_str = f"{combined_rotation[2]},{combined_rotation[0]},{-combined_rotation[1]}"
                
                elif tag_id == 1:
                    rod_vect_str = f"{-combined_rotation[0]},{-combined_rotation[2]},{-combined_rotation[1]}"

                elif tag_id == 2:
                    rod_vect_str = f"{-combined_rotation[2]},{combined_rotation[0]},{-combined_rotation[1]}"

                elif tag_id == 3:
                    rod_vect_str = f"{-combined_rotation[2]},{combined_rotation[0]},{-combined_rotation[1]}"

                elif tag_id == 4:
                    rod_vect_str = f"{-combined_rotation[2]},{combined_rotation[0]},{combined_rotation[1]}"

                elif tag_id == 5:
                    rod_vect_str = f"{combined_rotation[0]},{-combined_rotation[1]},{-combined_rotation[2]}"

                else:
                    rod_vect_str = f"{combined_rotation[0]},{combined_rotation[2]},{combined_rotation[1]}"
                
                # Send the Rodrigues vector via UDP
                sock.sendto(rod_vect_str.encode(), (udp_ip, udp_port))
                sent_udp = True  # Ensure only the first valid detection is sent

                
                
                euler = rodrigues_to_euler(rod_vect)
                euler2 = rodrigues_to_euler(combined_rotation)
                print(f"Detection ID: {tag_id}")
                print(f"X: {detection.pose.tvec[0]}, Y: {detection.pose.tvec[1]}, Z: {detection.pose.tvec[2]}")
                print(f"Angles: {combined_rotation}")
                print(f"Roll: {euler[0]*180/np.pi}, Pitch: {euler[1]*180/np.pi}, Yaw: {euler[2]*180/np.pi}")
                print(f"Roll: {euler2[0]*180/np.pi}, Pitch: {euler2[1]*180/np.pi}, Yaw: {euler2[2]*180/np.pi} -- Combined")
                sent_udp = True

        cv2.imshow("AprilTag Pose Estimation", frame_contig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_iteration = time.time()
        elapsed_time = iteration_beginning - end_iteration

    picam2.stop()
    cv2.destroyAllWindows()
    sock.close()  # Close the UDP socket when done

if __name__ == "__main__":
    main()
