import math
import h5py
import numpy as np
import os
import sys

def convert_h5_to_nnet():
    files = []
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = os.listdir("./")
    for ii in range(len(files)):
        fileName = files[ii]
        if fileName[-2:] == 'h5':
            print "File: " + fileName[:-2] + 'nnet'
            f = h5py.File(fileName, 'r')
            output_filepath = fileName[:-2] + 'nnet'
            output_filepath = output_filepath.replace('snapshots', 'nnet_files')
            with open(output_filepath, 'w') as f2:
                f2.write("Neural Network File Format by Kyle Julian, Stanford 2016\n")
                keys1 = f.keys()
                keys2 = f[keys1[0]].keys()
                numLayers = len(keys1)
                inputSize = len(f[keys1[0]][keys2[0]])
                outputSize = len(f[keys1[len(keys1) - 1]][keys2[1]])
                maxLayerSize = inputSize
                for key in keys1:
                    if len(f[key][keys2[1]]) > maxLayerSize:
                        maxLayerSize = len(f[key][keys2[1]])

                str = "%d,%d,%d,%d,\n" % (
                    numLayers, inputSize, outputSize, maxLayerSize)
                f2.write(str)
                str = "%d," % inputSize
                f2.write(str)
                for key in keys1:
                    str = "%d," % len(f[key][keys2[1]])
                    f2.write(str)

                f2.write("\n")
                if "DRL" not in fileName:
                    if "sym.h5" not in fileName:
                        print "Not Symmetric"
                        f2.write("0,\n")  # Symmetric Boolean
                        # Mininmum Input Values
                        f2.write("0.0,-3.141593,-3.141593,100.0,0.0,0.0,-3.0,\n")
                        # Maximum Input Values
                        f2.write("60760.0,3.141593,3.141593,800.0,800.0,100.0,3.0,\n")
                        # Means of inputs for normalizations
                        f2.write(
                            "1.9791091e+04,0.0,0.0,450.0,400.0,35.1111111,0.0,3.58839022945,\n")
                        # Ranges of inputs for normalizations
                        f2.write(
                            "60261.0,6.28318530718,6.28318530718,700.0,800.0,100.0,6.0,387.158972402,\n")
                    else:
                        print "Symmetric Neural Network!"
                        f2.write("1,\n")  # Symmetric Boolean
                        f2.write("0.0,-3.141593,0.0,100.0,0.0,0.0,-3.0,\n")
                        # Maximum Input Values
                        f2.write("60760.0,3.141593,3.141593,800.0,800.0,100.0,3.0,\n")
                        # Means of inputs for normalizations
                        f2.write(
                            "1.9791091e+04,0.0,1.570796326623,450.0,400.0,35.1111111,0.0,3.58839022945,\n")
                        # Ranges of inputs for normalizations
                        f2.write(
                            "60261.0,6.28318530718,3.141592653589,700.0,800.0,100.0,6.0,387.158972402,\n")
                else:
                    print "DRL!"
                    f2.write("0\n")  # Symmetric Boolean
                    # Mininmum Input Values
                    f2.write("0.0, -3.141593, -3.141593, 10.0, 10.0,\n")
                    # Maximum Input Values
                    f2.write("3000.0, 3.141593, 3.141593, 20.0, 20.0,\n")
                    # Means of inputs for normalizations
                    f2.write("1500.0,0.0,0.0,15.0,15.0,0.0\n")
                    # Ranges of inputs for normalizations
                    f2.write("3000.0,6.283185307,6.283185307,10.0,10.0,1.0\n")

                # Write weights and biases of neural network
                for key in f.keys():
                    for key2 in f[key].keys():
                        data = np.array(f[key][key2]).T
                        for i in range(len(data)):
                            for j in range(int(np.size(data) / len(data))):
                                str = ""
                                if int(np.size(data) / len(data)) == 1:
                                    # Four digits written. More can be used, but that
                                    # requires more more space.
                                    str = "%.4e," % data[i]
                                else:
                                    str = "%.4e," % data[i][j]
                                f2.write(str)
                            f2.write("\n")

if __name__ == '__main__':
    convert_h5_to_nnet()
