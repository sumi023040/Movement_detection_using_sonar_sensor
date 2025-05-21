import pandas as pd
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def view_data(f):
    for file in f:
        print(file)
        data = pd.read_csv(file)
        print(data.shape)
        print(data.sample(2))
        print('#####################################################')
        print('#####################################################')
        # print(data.sample(1))


def plot_data(f):

    for i, file in enumerate(f):
        data = pd.read_csv(file)

        d = data.sample(1).values
        d1 = data.sample(1).values
        d2 = data.sample(1).values
        d3 = data.sample(1).values

        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(d[0])
        axs[0,1].plot(d1[0])
        axs[1,0].plot(d2[0])
        axs[1,1].plot(d3[0])
        plt.title(file)
        plt.show()


def loading(base):
    base_path = os.path.join(base, "Data/")
    side = []
    facing = []
    nomove = []
    signal_folders = []

    signal_folders.append(os.path.join(base_path, "facing"))
    signal_folders.append(os.path.join(base_path, "side"))
    signal_folders.append(os.path.join(base_path, "nomove"))

    for folder in signal_folders:
        file_list = os.listdir(folder)
        fol_char = folder.split("/")
        if fol_char[-1] == "facing":
            for f in file_list:
                facing.append(os.path.join(folder, f))
        elif fol_char[-1] == "side":
            for f in file_list:
                side.append(os.path.join(folder, f))
        else:
            for f in file_list:
                nomove.append(os.path.join(folder, f))

    signla_dict = {'side':side,
                   'facing':facing,
                   'nomove':nomove}
    
    return signla_dict



def main():
    load_dotenv()
    base = os.getenv("BASE_PATH")
    files = loading(base)
    x = list(files.values())
    
    # print(x[0])
    # print(x[1])
    # print(x[2])

    if x[0]:
        view_data(x[0])
        plot_data(x[0])

    if x[1]:
        view_data(x[1])
        plot_data(x[1])

    if x[2]:
        view_data(x[2])
        plot_data(x[2])

    # view_data(base)

if __name__ == "__main__":
    main()
