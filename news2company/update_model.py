import os
import argparse
import zipfile

from config import transfer_config as tc
from pipeline.transfer import TransferManager


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload pre-trained NER model to S3.")
    parser.add_argument(
        '-m', '--model_path',
        help="Dir of model",
        type=str
    )
    parser.add_argument(
        '-z', '--zipped_model',
        help="Zipped model dir, model.zip for example",
        type=str
    )
    args = parser.parse_args()

    manager = TransferManager(bucket=tc['bucket'])
    model_key = tc['model']['upstream'].rstrip('/') + '/model.zip'
    if args.zipped_model:
        manager.upload_fileobj(args.zipped_model, model_key)
    elif args.model_path:
        with zipfile.ZipFile('model.zip', 'w', zipfile.ZIP_DEFLATED) as zip_f:
            zipdir(args.model_path, zip_f)
        manager.upload_fileobj('model.zip', model_key)
        os.remove('model.zip')
