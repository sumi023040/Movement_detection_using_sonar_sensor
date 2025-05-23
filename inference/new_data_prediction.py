import os
import pandas as pd
import numpy as np
from scipy.signal import bessel, filtfilt, find_peaks
from scipy.fft import fft
from scipy.stats import skew, kurtosis, entropy
from dotenv import load_dotenv
import joblib


def apply_filter(base, data):
    b, a = bessel(N=2, Wn=0.6, btype='low', analog=False, norm='phase')
    signal_filterred = []
    for ind, item in data.iterrows():
        converted_item = item.values.astype(float)
        signal_filterred.append(filtfilt(b, a, converted_item))
    filterred_df = pd.DataFrame(signal_filterred)
    output_path = os.path.join(base, 'inference/', 'filtered.csv' )
    filterred_df.to_csv(output_path, index=False)

    return filterred_df


def getting_first_peak(base, data):
    first_peak_loc = []
    for ind, signals in data.iterrows():
        signals = signals.values.astype(float)
        if np.all(np.isnan(signals)) or len(signals) == 0:
            first_peak_loc.append(np.nan)
            continue

        signals = np.nan_to_num(signals)

        std = np.std(signals)
        peaks, properties = find_peaks(signals, height=(std * 0.2))  # , prominence=0.28)

        if len(peaks) > 0:
            first_peak_loc.append(peaks[0])
        else:
            first_peak_loc.append(np.nan)

    data['First_peak'] = first_peak_loc
    data.dropna(inplace=True)
    output_path = os.path.join(base, 'inference/', 'filtered.csv')
    data.to_csv(output_path, index=False)
    return data


def getting_fft(base, data):
    fs = 1953125
    output_path = os.path.join(base, 'inference/', "fft.csv")

    signal_array = data.to_numpy(dtype=float)
    N = signal_array.shape[1]
    freqs = np.fft.fftfreq(N, d=1 / fs)[:N // 2]
    column_names = [f"{round(f, 2)}Hz" for f in freqs]

    fft_data = []
    for signal in signal_array:
        spectrum = np.abs(fft(signal))[:N // 2]
        fft_data.append(spectrum)

    fft_df = pd.DataFrame(fft_data, columns=column_names)

    fft_df.to_csv(output_path, index=False)

    return fft_df


def getting_statistical_features(base, data):
    all_feature_rows = []
    data.dropna(inplace=True)
    peaks = data['First_peak'].copy()

    data.drop(columns=['First_peak'], inplace=True)

    for idx, row in data.iterrows():
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
            "percentile_75": np.percentile(signal, 75)
        }

        chunk_size = 10000
        moving_features = {}

        for i in range(0, len(signal), chunk_size):
            chunk = signal[i:i + chunk_size]
            moving_features[f'avg_in_{i + 1}st_10k'] = np.round(np.mean(chunk).item(), 5)
            moving_features[f'std_in_{i + 1}st_10k'] = np.round(np.std(chunk).item(), 5)

        full_row = {**moving_features, "First_peak": peaks.loc[idx], **base_features}
        all_feature_rows.append(full_row)

    feature_df = pd.DataFrame(all_feature_rows)
    output_path = base + 'inference/statistical_feature.csv'
    feature_df.to_csv(output_path, index=False)

    return feature_df


def getting_frequency_features(base, fft_df):
    fft_features = []

    fft_df = fft_df.reindex(sorted(fft_df.columns, key=lambda x: float(x.replace("Hz", ""))), axis=1)
    freqs = np.array([float(col.replace("Hz", "")) for col in fft_df.columns])

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
            'flatness': flatness
        })

    fft_feature = pd.DataFrame(fft_features)
    output_path = base + 'inference/fft_feature.csv'
    fft_feature.to_csv(output_path, index=False)

    return fft_feature


def combine_all_features(freq_features, stat_features, base):
    fft = pd.read_csv(freq_features)
    norm = pd.read_csv(stat_features)
    col_name = fft.columns

    for col in col_name:
        norm[col] = fft[col]

    norm.to_csv(base + 'inference/all_features.csv')

    return norm


def main():
    load_dotenv()
    base = os.getenv('BASE_PATH')
    data = pd.read_csv()

    filtered = apply_filter(base, data)

    data = filtered.apply(lambda row: row.clip(lower=row.mean()), axis=1)
    data.to_csv(filtered, index=False)

    data = getting_first_peak(base, data)

    fft_data = getting_fft(base, data)

    statistical_feature = getting_statistical_features(base, data)

    frequency_features = getting_frequency_features(base, fft_data)

    feature_dataset = combine_all_features(statistical_feature, frequency_features, base)

    # svm_model = joblib.load('multi_class_svm.joblib')

    mlp_model = joblib.load('mlp_model.joblib')

    predictions = mlp_model.predict(feature_dataset)

    # HERE BASED ON THE VALUE EITHER THE SENSOR LIGHT CAN BE TURNED ON OR OFF.
    # FOR DEMONSTRATION PURPOSE IT IS JUST PRINTING THE VAL.
    # THIS VAL HAVE TO BE SENT TO THE API FOR LIGHT SWITCH
    for val in predictions:
        if val == 0:
            print(f'Movement is type {val}')
        elif val == 1:
            print(f'Movement is type {val}')
        else:
            print(f'Movement is type {val}')


if __name__ == '__main__':
    main()
