import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.multimodal_preprocessors import IMUPreprocessor
import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def get_embeddings(extrapolated_imu_data, device, out_file):
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    N = extrapolated_imu_data.shape[0]  # Number of data points
    final_embedding = np.array([[]])
    # change for different dataset
    for i in range(0, 5):
        print(i*9, i*9+6)
        # change for different dataset
        time_series = extrapolated_imu_data[:,:,i*9:i*9+6].astype(np.float32)
        # time_series = np.expand_dims(np.transpose(time_series), axis=0)
        # change for different dataset
        time_series = np.moveaxis(time_series, 1, 2)
        print('time_series shape: ', time_series.shape)
        # Load data
        inputs = {
            ModalityType.IMU: torch.from_numpy(time_series).to(device),
        }
        with torch.no_grad():
            embeddings = model(inputs)

        imu_embedding = (embeddings[ModalityType.IMU].to("cpu").numpy())
        print('imu_embedding shape: ', imu_embedding.shape)
        # f=open(out_file,'a')
        # np.savetxt(out_file, imu_embedding, fmt='%f')
        if i == 0:
            final_embedding = imu_embedding
        else:
            final_embedding = np.concatenate((final_embedding, imu_embedding))

        print('final_embedding shape: ', final_embedding.shape)
        
    np.save(out_file, final_embedding)

def extrapolate_timeseries(imu_data_path):
    # preprocessor is already loaded into the forward function
    # only need to write the load_and_transform function

    original_data = np.load(imu_data_path)

    # Simulating a 3D NumPy array with shape (N, 150, s)
    N = original_data.shape[0]  # Number of data points
    time_series_length = original_data.shape[1]
    s = original_data.shape[2]   # Number of sensors

    # Desired number of points for extrapolation
    desired_points = 2000

    # Create an array to store the extrapolated data
    extrapolated_data = np.empty((N, desired_points, s))

    # Calculate the interpolation indices
    indices = np.linspace(0, time_series_length - 1, desired_points)

    # Loop through data points and sensors
    for i in range(N):
        for j in range(s):
            original_time_series = original_data[i, :, j]

            # Normalize the original_time_series from -1 to 1
            # normalized_time_series = np.interp(original_time_series, (original_time_series.min(), original_time_series.max()), (-1, +1))
            # normalized_time_series = original_time_series

            extrapolated_data[i, :, j] = np.interp(indices, np.arange(time_series_length), original_time_series)
    
    plt.subplot(2, 1, 1)
    plt.plot(original_data[0,:,1], label='Original Data', marker='o')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    # Plot the extrapolated data
    plt.subplot(2, 1, 2)
    plt.plot(extrapolated_data[0,:,1], label='Extrapolated Data', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    # print(original_data.shape)
    # print(extrapolated_data.shape)

    # print(original_data[0,:,2].shape)
    # print(extrapolated_data[0,:,2].shape)
    # print(original_data[0,:,2])
    # print(extrapolated_data[0,:,2])

    plt.tight_layout()  # Adjust subplots for better spacing

    plt.savefig("extrapolated_data_plot_for_all.png")
    return extrapolated_data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        assert False, 'Usage: python imu_embedding.py relative_data_path_to_npy_file out_file_name'

    print(sys.argv)
    data_path = sys.argv[1]
    out_file = sys.argv[2]
    extrapolated_data = extrapolate_timeseries(data_path)

    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    get_embeddings(extrapolated_data, device, out_file)


    # original_data = dataset[0,:,0]
    # print(x.shape)
    # plt.plot(x)
    # plt.savefig("line_plot.png")

    # Number of points you want to interpolate to (e.g., 1000 points)
    # desired_points = 2000

    # # Calculate the interpolation indices
    # indices = np.linspace(0, len(original_data) - 1, desired_points)

    # # Perform linear interpolation
    # extrapolated_data = np.interp(indices, np.arange(len(original_data)), original_data)

    # # Create a figure for the plots
    # plt.figure(figsize=(12, 6))

    # Plot the original data
    
