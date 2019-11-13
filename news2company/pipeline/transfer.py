import os
import boto3


class TransferManager:
    def __init__(self, bucket):
        self.client = boto3.client('s3')
        self.bucket = bucket

    def upload_file(self, filename, key):
        self.client.upload_file(filename, self.bucket, key)

    def upload_fileobj(self, filename, key):
        with open(filename, 'rb') as data:
            self.client.upload_fileobj(data, self.bucket, key)

    def download_file(self, key, filename):
        self.client.download_file(self.bucket, key, filename)

    def download_fileobj(self, key, filename):
        with open(filename, 'wb') as f:
            self.client.download_fileobj(self.bucket, key, f)

    def download_dir_file(self, dir_key, dir_name):
        """
        download file in 'dir' from s3, 'dir' in 'dir' will not be download
        :param dir_key: string, dir key on s3, endswith '/'
        :param dir_name: string, local dir name, not endswith os.sep('/' or '\\'),
            Note this will become confused if the path elements to create include pardir (eg. “..” on UNIX systems).
            Refer to doc of os.makedirs()
        :return: None
        """
        sep = os.sep
        if dir_key[-1] != '/':
            dir_key = dir_key + '/'
        if dir_name[-1] == sep:
            dir_name = dir_name[:-1]

        res = self.client.list_objects(Bucket=self.bucket, Prefix=dir_key, Delimiter='/')
        keys = []
        for contents in res['Contents']:
            key = contents['Key']
            keys.append(key)
        if len(keys) == 0:
            raise FileNotFoundError("No such dir on s3: {0}".format(dir_key))

        os.makedirs(dir_name, exist_ok=True)

        for key in keys:
            filename = key.split('/')[-1]
            self.download_file(key, os.path.join(dir_name, filename))
