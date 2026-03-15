import argparse
import os
import shutil

##############
# Constants
##############
SOURCE_DIR = '.data'
'''The dataset should be downloaded in SOURCE_DIR'''

DEST_DIR = SOURCE_DIR
'''Output directory for train/test split'''

DEST_TRAIN = f'{DEST_DIR}/train/images'
DEST_TEST = f'{DEST_DIR}/test/images'

LABELS = ['glide', 'loose', 'none', 'slab']


if __name__ == '__main__':
    '''Generate train/test split for the image classification task'''

    parser = argparse.ArgumentParser(
        description='Generate train/test image split folders from split text files.')
    parser.add_argument(
        '--source-dir',
        type=str,
        default=SOURCE_DIR,
        help='Dataset root containing train.txt, test.txt and images/.',
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    dest_dir = source_dir
    dest_train = os.path.join(dest_dir, 'train', 'images')
    dest_test = os.path.join(dest_dir, 'test', 'images')

    # Check that train/test split txt files exist
    train_split = os.path.join(source_dir, 'train.txt')
    test_split = os.path.join(source_dir, 'test.txt')
    assert os.path.exists(train_split), f'File not found: {train_split}'
    assert os.path.exists(test_split), f'File not found: {test_split}'

    # Read image paths for train/test split
    with open(train_split) as f:
        train_ims = [line.rstrip('\n') for line in f]
    with open(test_split) as f:
        test_ims = [line.rstrip('\n') for line in f]

    # Delete and re-create output directories
    for train_test in [dest_train, dest_test]:
        try:
            shutil.rmtree(train_test)
        except:
            pass
        for l in LABELS:
            os.makedirs(os.path.join(train_test, l), exist_ok=False)

    # Copy images into destination directory
    print('Generating train/test split...')
    ims_directory = os.path.join(source_dir, 'images')
    for root, dirnames, filenames in os.walk(ims_directory):
        for file_name in filenames:
            assert file_name.lower().endswith(
                '.jpg'), f'Unexpected file name {file_name}'

            # Get whether file is in the train or test split
            path_tail = '/' + root.split('/')[-1] + f'/{file_name}'
            target_dir = None
            if path_tail in train_ims:
                target_dir = dest_train
            elif path_tail in test_ims:
                target_dir = dest_test
            else:
                raise FileNotFoundError(
                    f'Image {path_tail} not in train or test split')

            # Copy image to destination directory
            dest_file = target_dir + path_tail
            shutil.copyfile(f'{root}/{file_name}', dest_file)

    print('Done')
