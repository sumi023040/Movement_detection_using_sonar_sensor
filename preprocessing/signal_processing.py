import os
import gc
import re
from data_loading import loading
import pandas as pd
import numpy as np
from scipy.signal import bessel, filtfilt, welch, find_peaks
from scipy.fft import fft
from dotenv import load_dotenv
import matplotlib.pyplot as plt


def applying_filter(base, files):
    b, a = bessel(N=2, Wn=0.6, btype='low', analog=False, norm='phase')
    for signal_class in list(files.keys()):
        signal_filterred = []
        
        for file in files[signal_class]:
            path_char = file.split('/')
            signal_data = pd.read_csv(file)
            print(path_char[-1])
            print('##################')
            for ind, item in signal_data.iterrows():
                converted_item = item.values.astype(float)
                signal_filterred.append(filtfilt(b, a, converted_item))
            filterred_df = pd.DataFrame(signal_filterred)
            output_path = os.path.join(base, 'Data', signal_class, f"filtered_{path_char[-1]}")
            filterred_df.to_csv(output_path, index=False)

            del signal_data
            del converted_item   
            del filterred_df
            gc.collect()
            print('$$$$$$$$$$$$$$$$$$')
        del signal_filterred 


def getting_fft(base, files):
    fs = 1953125
    for signal_class, file_list in files.items():
        output_path = os.path.join(base, 'Data', signal_class, "fft.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            os.remove(output_path)
        header_written = False

        for file in file_list:
            print(f'Processing: {file}')
            chunk_iter = pd.read_csv(file, chunksize=1000)
            for chunk in chunk_iter:
                signal_array = chunk.to_numpy(dtype=float)
                N = signal_array.shape[1]
                freqs = np.fft.fftfreq(N, d=1 / fs)[:N // 2]
                column_names = [f"{round(f, 2)}Hz" for f in freqs]

                fft_data = []
                for signal in signal_array:
                    spectrum = np.abs(fft(signal))[:N // 2]
                    fft_data.append(spectrum)

                fft_df = pd.DataFrame(fft_data, columns=column_names)

                # Append to fft.csv
                fft_df.to_csv(output_path, mode='a', header=not header_written, index=False)
                header_written = True  # Only write header once

                # Clean up
                del fft_data, fft_df, signal_array
                gc.collect()

            print(f'Done with file: {file}')
        print(f'Finished folder: {signal_class}')


def removing_below_row_mean(files):
    for signal_class in list(files.keys()):
        for file in files[signal_class]:
            data = pd.read_csv(file)
            data = data.apply(lambda row: row.clip(lower=row.mean()), axis=1)

            data.to_csv(file, index=False)


def first_peak(files):
    # plt.figure()
    for signal_class in list(files.keys()):

        for file in files[signal_class]:
            data = pd.read_csv(file)
            print(file)
            print('#################################')
            first_peak_loc = []
            for ind, signals in data.iterrows():
                signals = signals.values.astype(float)
                if np.all(np.isnan(signals)) or len(signals) == 0:
                    first_peak_loc.append(np.nan)
                    continue

                signals = np.nan_to_num(signals)

                std = np.std(signals)
                peaks, properties = find_peaks(signals, height=(std * 0.2))#, prominence=0.28)

                if len(peaks) > 0:
                    first_peak_loc.append(peaks[0])
                else:
                    first_peak_loc.append(np.nan)

                # plt.plot(signals, color='grey')
                # plt.plot(peaks, signals[peaks], color='red')
                # plt.show()
            data['First_peak'] = first_peak_loc
            data.dropna(inplace=True)
            print(file)
            data.to_csv(file, index=False)
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

            del data
            del signals
            gc.collect()


def main():
    load_dotenv()
    base = os.getenv('BASE_PATH')
    files = loading(base)

    applying_filter(base, files)

    getting_fft(base, files)

    removing_below_row_mean(files)

    first_peak(files)


if __name__=='__main__':
    main()
