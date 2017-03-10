import numpy as np
import os


class DataReader(object):

    def __init__(self, input_dir, output_dir, num_steps=10, name=None):
        print(name + " data reader initialization...")
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._input_file_list = sorted(os.listdir(input_dir))
        self._output_file_list = sorted(os.listdir(output_dir))
        self._file_len = len(self._input_file_list)
        self._name = name
        assert self._file_len == len(self._output_file_list), "# input files and output file is not matched"

        self._num_file = 0
        self._num_steps = num_steps
        self._start_idx = 0
        self._inputs = self._read_input(self._input_file_list[self._num_file])
        self._outputs = self._read_output(self._output_file_list[self._num_file])

        assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
            ("# samples is not matched between input: %d and output: %d files"
             % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))

        self.num_samples = np.shape(self._outputs)[0]

        print("Done.")
        print("Loaded " + self._name + " file number : %d" % (self._num_file + 1))
        # stft_mag, stft_phase = np.load(file_list[0])

    def _read_input(self, input_file_dir):

        num_steps = self._num_steps
        data = np.load(self._input_dir+'/'+input_file_dir)  # data shape : (channel, # total frame, fft_size)
        data = np.transpose(data, (1, 2, 0))  # data shape : (# total frame, fft_size, channel)

        data_len = data.shape[0] - (data.shape[0] % num_steps)
        data = data[0:data_len, :]

        fft_size = data.shape[1]
        channel = data.shape[-1]
        data = data.reshape(-1, num_steps, fft_size, channel)

        return data

    def _read_output(self, output_file_dir):

        num_steps = self._num_steps
        data = np.load(self._output_dir+'/'+output_file_dir)  # data shape : (# total frame,)
        data = data.reshape(-1, 1)  # data shape : (# total frame, 1)

        data_len = data.shape[0] - (data.shape[0] % num_steps)
        data = data[0:data_len, :]
        data = data.reshape(-1, num_steps, 1)

        return data

    def next_batch(self, batch_size):

        if self._start_idx + batch_size > self.num_samples:

            self._start_idx = 0
            self._num_file += 1

            print("No more new data in this " + self._name + " file. try to read other file to make a mini-batch")
            if self._num_file > self._file_len - 1:
                self._num_file = 0
                print("No more new file. try to return to first " + self._name + " file to make a mini-batch")

            self._inputs = self._read_input(self._input_file_list[self._num_file])
            self._outputs = self._read_output(self._output_file_list[self._num_file])

            assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
                ("# samples is not matched between input: %d and output: %d files"
                 % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))

            self.num_samples = np.shape(self._outputs)[0]
            # print("current file number : %d, samples : %d" % (self._num_file + 1, self.num_samples))
            print("Loaded " + self._name + " file number : %d" % (self._num_file + 1))

        inputs = self._inputs[self._start_idx:self._start_idx + batch_size, :, :, 0]
        outputs = self._outputs[self._start_idx:self._start_idx + batch_size, :, :]

        self._start_idx += batch_size
        # print(self._start_idx)
        # print(self.num_samples)
        return inputs, outputs

        #num_batches = (np.shape(self._outputs)[0] - np.shape(self._outputs)[0] % batch_size) / batch_size


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# file_dir = "/home/sbie/github/VAD_KJT/Data/data_2017_0302/Aurora2withSE"
# input_dir1 = file_dir + "/Noisy_Aurora_STFT_npy"
# output_dir1 = file_dir + "/labels"
# dr = DataReader(input_dir1, output_dir1, num_steps=30)
#
#
# for i in range(1000000):
#     tt , pp = dr.next_batch(500)
#     cc=dense_to_one_hot(pp.reshape(-1,1))
#     print("asdf")


