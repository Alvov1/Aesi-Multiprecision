import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import numpy as np
import json

def parse_json(path: str):
    with open(path) as file:
        tests = json.load(file)["benchmarks"]

        functions = dict()
        for test in tests:
            function, library = test["name"].split('_')
            if function not in functions:
                functions[function] = dict()
            functions[function][library] = int(test["real_time"])

        records = dict()
        for _, libraries in functions.items():
            for library in libraries:
                if library not in records:
                    records[library] = list()
                records[library].append(libraries[library])

        return list(functions.keys()), records


def plot(operations, data, output_path, font):
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = font
    plt.rcParams['font.weight'] = 'bold'

    plt.style.use('fivethirtyeight')
    libraries = list(data.keys())
    values = np.array(list(data.values()))
    x = np.arange(len(operations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, lib in enumerate(libraries):
        ax.bar(x + i * width, values[i], width, label=lib, color=colors[i])

    ax.set_xlabel('Operations', fontsize=15, weight='bold')
    ax.set_ylabel('Execution Time (ns)', fontsize=15, weight='bold')
    ax.set_title('Execution Time Comparison', fontsize=16, weight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(operations, rotation=45, ha='right', fontsize=13)
    ax.set_yscale('log')
    ax.legend(title='Libraries', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == '__main__':
    measures = sys.argv[1]
    output_image_path = sys.argv[2]
    print(f"Loading measures table from '{measures}'")
    print(f"Output image location '{output_image_path}'")

    font_path, font_name = ('CourierPrime-Bold.ttf', 'Courier Prime')
    fm.fontManager.addfont(font_path)

    test_names, records = parse_json(measures)
    plot(test_names, records, output_image_path, font_name)