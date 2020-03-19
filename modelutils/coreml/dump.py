import argparse
import os
import coremltools


def load_model(filename):
    return coremltools.models.MLModel(filename).get_spec()


def dump(filename, save_to_file):
    model = load_model(filename)
    if save_to_file:
        save_filename = filename + '.dump'
        if os.path.exists(save_filename):
            raise RuntimeError(f"{save_filename} already exists.")
        with open(save_filename, 'w') as f:
            f.write(str(model))
    else:
        print(model)


def main():
    parser = argparse.ArgumentParser('Dump a CoreML model to text')
    parser.add_argument('model_filename', type=str, help='Filename for the input CoreML file')
    parser.add_argument('-s', '--save', action='store_true', help='Save to .dump file')

    args = parser.parse_args()
    dump(args.model_filename, args.save)


if __name__ == '__main__':
    main()
