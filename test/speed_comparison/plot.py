import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
import numpy as np


def parse_xml_and_plot(xml_file, output, version):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    test_names = []
    cryptopp_times = []
    gmp_times = []
    aesi_times = []

    for testsuite in root.findall('testsuite'):
        suite_name = testsuite.get('name')
        test_names.append(suite_name)

        for testcase in testsuite.findall('testcase'):
            name = testcase.get('name')
            time = float(testcase.get('time'))
            if name == 'CryptoPP':
                cryptopp_times.append(time)
            elif name == 'GMP':
                gmp_times.append(time)
            elif name == 'Aesi':
                aesi_times.append(time)

    x = np.arange(len(test_names))
    width = 0.25

    fig, ax = plt.subplots()
    rects3 = ax.bar(x + width, aesi_times, width, capstyle='butt', label='Aesi')
    rects1 = ax.bar(x - width, cryptopp_times, width, capstyle='projecting', label='CryptoPP')
    rects2 = ax.bar(x, gmp_times, width, capstyle='round', label='GMP')

    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Execution Time {version}\n(generated automatically)')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names)
    ax.legend()

    fig.tight_layout()
    plt.savefig(output, dpi=300)


if __name__ == '__main__':
    measures = sys.argv[1]
    output_image_path = sys.argv[2]
    version = sys.argv[3]
    print(f"Loading measures table from '{measures}'")
    print(f"Output image location '{output_image_path}'")
    print(f"Version '{version}'")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    parse_xml_and_plot(measures, output_image_path, version)

