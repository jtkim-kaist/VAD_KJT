import numpy as np
import math
import os


def data_concat(input_dir, output_dir, input_save_dir, output_save_dir, num_concat):
    input_file_list = sorted(os.listdir(input_dir))
    output_file_list = sorted(os.listdir(output_dir))
    file_len = len(input_file_list)

    assert file_len == len(output_file_list)
    num_files = math.ceil(file_len / num_concat)

    for i in range(num_files):
        print("make %d / %d file" % (i+1, num_files))
        if i == (num_files - 1):  # last file
            for j in range(file_len % num_concat):
                if j == 0:
                    input_file = np.load(input_dir+'/'+input_file_list[i*num_concat+j]) # data shape : (channel, # total frame, fft_size)
                    input_concat_file = input_file

                    output_file = np.load(output_dir+'/'+output_file_list[i*num_concat+j])
                    output_concat_file = output_file

                else:
                    input_file = np.load(input_dir+'/'+input_file_list[i*num_concat+j])
                    input_concat_file = np.concatenate((input_concat_file, input_file), axis=1)

                    output_file = np.load(output_dir+'/'+output_file_list[i*num_concat+j])
                    output_concat_file = np.concatenate((output_concat_file, output_file))
        else:
            for j in range(num_concat):
                if j == 0:
                    input_file = np.load(input_dir+'/'+input_file_list[i*num_concat+j]) # data shape : (channel, # total frame, fft_size)
                    input_concat_file = input_file

                    output_file = np.load(output_dir+'/'+output_file_list[i*num_concat+j])
                    output_concat_file = output_file
                else:
                    input_file = np.load(input_dir+'/'+input_file_list[i*num_concat+j])
                    input_concat_file = np.concatenate((input_concat_file, input_file), axis=1)

                    output_file = np.load(output_dir+'/'+output_file_list[i*num_concat+j])
                    output_concat_file = np.concatenate((output_concat_file, output_file))
        np.save(input_save_dir + '/Noisy_Aurora' + str(i+1).zfill(2), input_concat_file)
        np.save(output_save_dir + '/Clean_Aurora' + str(i+1).zfill(2), output_concat_file)


# aa = np.load("/home/sbie/github/VAD_KJT/Data/data_0308_2017/Aurora2withSE/Noisy_Aurora_STFT_npy/Noisy_Aurora01.npy")
# bb = np.load("/home/sbie/github/VAD_KJT/Data/data_0308_2017/Aurora2withSE/labels/Clean_Aurora01.npy")
#
# print("sadfsafd")
file_dir = "/home/sbie/github/VAD_KJT/Data/data_0302_2017"
input_dir2= file_dir + "/noiseXed/01/10"
output_dir2 = file_dir + "/label/NX_Aurora2"

save_dir2 = "/home/sbie/github/VAD_KJT/Data/data_0308_2017/Aurora2withNX"
input_dir3 = save_dir2 + "/Noisy_Aurora_STFT_npy/Babble/SNR_10"
output_dir3 = save_dir2 + "/labels"

data_concat(input_dir2, output_dir2, input_dir3, output_dir3, 1001)

