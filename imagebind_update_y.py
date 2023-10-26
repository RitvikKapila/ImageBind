import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        assert False, 'Usage: python imagebind_update_y.py relative_path_to_npy_file out_file_name'

    print(sys.argv)
    
    label_file_path = sys.argv[1]
    out_file = sys.argv[2]
    
    y = np.load(label_file_path)
    print(y.shape)
    print(y)

    # change this repeat frequency for other datasets
    y = np.concatenate((y, y, y, y, y))

    np.save(out_file, y)
