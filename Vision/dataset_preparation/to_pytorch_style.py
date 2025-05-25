from argparse import ArgumentParser
import os


def main(split_path):

    # load up the relevant files
    images_path = os.path.join(split_path, 'images')
    annotations_file = [x for x in os.listdir(split_path) if 'annotations' in x][0]
    annotations_file_path = os.path.join(split_path, annotations_file)

    with open(file=annotations_file_path, mode='r') as file:
        for i, line in enumerate(file):
            # print(f'{i+1}')
            img_name, class_id = line.split(' ')
            class_id = class_id[:-1]
            img_name = f"ILSVRC2011_val_{img_name}.JPEG"
            
            # make class id folder
            class_path = os.path.join(split_path, class_id)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            # move image file
            try:
                from_path = os.path.join(images_path, img_name)
                to_path = os.path.join(class_path, img_name)

                os.rename(from_path, to_path)
            except Exception as e:
                print(f'skipping... {from_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-split_path', help='/path/to/split (test, train, val). Folder must have an images folder and annotations.txt')

    args = parser.parse_args()

    main(args.split_path)
