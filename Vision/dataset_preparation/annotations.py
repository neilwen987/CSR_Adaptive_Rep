import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--xml_dir', type=str, help='Path to the directory containing .xml files')
parser.add_argument('--output_file', type=str, default='annotations.txt', help='Output file name')
args = parser.parse_args()

def main():
    with open(args.output_file, 'w') as out_f:
        for filename in os.listdir(args.xml_dir):
            if filename.endswith('.xml'):
                file_path = os.path.join(args.xml_dir, filename)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    full_filename = root.findtext('filename', default='')
                    image_id = os.path.splitext(full_filename)[0].split('_')[-1]

                    obj = root.find('object')
                    if obj is not None:
                        name = obj.findtext('name', default='unknown')
                        out_f.write(f"{image_id} {name}\n")
                except ET.ParseError as e:
                    print(f"Unable to identify: {file_path}, Error: {e}")

if __name__ == "__main__":
    main()
