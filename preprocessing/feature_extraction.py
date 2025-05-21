import os
import matplotlib.pyplot as plt
from data_loading import loading
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.stats import entropy, skew, kurtosis
import gc


def filtered_signal_feature_extraction(base, signal_file):
    for signal_class in list(signal_file.keys()):
        all_feature_rows = []
        for file_path in signal_file[signal_class]:
            print(f"Processing: {file_path}")
            df = pd.read_csv(file_path)

            df.dropna(inplace=True)
            peaks = df['First_peak'].copy()

            df.drop(columns=['First_peak'], inplace=True)

            for idx, row in df.iterrows():
                signal = pd.to_numeric(row.values, errors='coerce')
                signal = signal[~np.isnan(signal)]

                if len(signal) == 0:
                    continue

                base_features = {
                    "mean": np.mean(signal),
                    "std": np.std(signal),
                    "rms": np.sqrt(np.mean(signal ** 2)),
                    "skewness": skew(signal),
                    "kurtosis": kurtosis(signal),
                    "energy": np.sum(signal ** 2),
                    "percentile_25": np.percentile(signal, 25),
                    "percentile_50": np.percentile(signal, 50),
                    "percentile_75": np.percentile(signal, 75),
                    "label": signal_class,
                }

                chunk_size = 10000
                moving_features = {}

                for i in range(0, len(signal), chunk_size):
                    chunk = signal[i:i + chunk_size]
                    moving_features[f'avg_in_{i+1}st_10k'] = np.round(np.mean(chunk).item(), 5)
                    moving_features[f'std_in_{i+1}st_10k'] = np.round(np.std(chunk).item(), 5)

                full_row = {**moving_features, "First_peak": peaks.loc[idx], **base_features}
                all_feature_rows.append(full_row)

        feature_df = pd.DataFrame(all_feature_rows)
        output_path = base + f'{signal_class}.csv'
        feature_df.to_csv(output_path, index=False)
        print(feature_df.shape)
    # Clean up memory
    del all_feature_rows
    gc.collect()


def adding_files(base, signal_file):
    combined = []
    for file in signal_file:
        one = pd.read_csv(file)
        combined.append(one)

    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(base+'feature.csv', index=False)


def extract_frequency_features(base, fft_files):
    fs = 1953125
    fft_features = []
    for signal_class in fft_files.keys():
        for file in fft_files[signal_class]:
            fft_df = pd.read_csv(file)
            # Ensure columns are sorted by frequency
            fft_df = fft_df.reindex(sorted(fft_df.columns, key=lambda x: float(x.replace("Hz", ""))), axis=1)
            freqs = np.array([float(col.replace("Hz", "")) for col in fft_df.columns])
            N = len(freqs)

            for _, row in fft_df.iterrows():
                spectrum = row.values.astype(float)
                total_power = np.sum(spectrum)
                if total_power == 0:
                    total_power = 1e-10

                norm_spectrum = spectrum / total_power

                dominant_freq = freqs[1:][np.argmax(spectrum[1:])]
                mean_freq = np.sum(freqs * norm_spectrum)
                band_mask = (freqs >= 0) & (freqs <= 50)
                band_energy = np.sum(spectrum[band_mask])
                spec_entropy = entropy(norm_spectrum)
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)

                fft_features.append({
                    'dominant_freq': dominant_freq,
                    'mean_freq': mean_freq,
                    'band_energy_0_50Hz': band_energy,
                    'spectral_entropy': spec_entropy,
                    'frequency_centroid': centroid,
                    'flatness': flatness,
                    'label': signal_class
                })
            print(f"{file} is complete")
    fft_feature = pd.DataFrame(fft_features)
    output_path = base + 'Data/fft_feature.csv'
    fft_feature.to_csv(output_path, index=False)

    print("Extracted features")


def add_frequency_features(base, files):
    freq_features = []
    for signal_type in files.keys():
        for file in files[signal_type]:
            data = pd.read_csv(file)
            print(file)
            print(data.shape)
            print('####################################')
    #         freq_features.append(data)
    #
    # freq_features = pd.concat(freq_features, ignore_index=True)
    # freq_features.to_csv(base+'freq_features.csv', index=False)

def combined_features(base, fft, norm):
    fft = pd.read_csv(fft)
    norm = pd.read_csv(norm)
    col_name = fft.columns

    for col in col_name:
        if col == 'label':
            continue
        else:
            norm[col] = fft[col]
    norm.to_csv(base+'all_features.csv')


def main():
    load_dotenv()
    base = os.getenv("BASE_PATH")
    files = loading(base)

    filtered_signal_feature_extraction(base + 'Data/', files)

    extract_frequency_features(base, files)

    files = [base + 'Data/side.csv', base + 'Data/nomove.csv', base + 'Data/facing.csv']
    adding_files(base + 'Data/', files)

    fft_feature = base+'Data/fft_feature.csv'
    feature = base+'Data/feature.csv'
    combined_features(base, fft_feature, feature)

    # add_frequency_features(base, files)

if __name__ == "__main__":
    main()
