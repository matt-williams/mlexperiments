from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
import os
import stat

class GDrive:
    def __init__(self):
        self.authenticate()

    def authenticate(self):
        try:
            from google.colab import auth
            auth.authenticate_user()
            gauth = GoogleAuth()
            gauth.credentials = GoogleCredentials.get_application_default()
        except ImportError:
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()
        return gauth

    def load(self, files, dest_path='', force=True, required=True, permissions=None):
        gauth = self.authenticate()
        drive = GoogleDrive(gauth)
        files = [files] if not isinstance(files, (list, tuple)) else files
        for file in files:
            if not os.path.exists(file) or force:
                dir = drive.ListFile({'q': "title = '{}'".format(file)}).GetList()
                if len(dir) > 0:
                    dir[0].GetContentFile(dest_path + file)
                    if permissions:
                        os.chmod(dest_path + file, permissions)
                    print("Got {}".format(file))
                elif required:
                    raise Exception("File {} missing from Google Drive".format(file))

    def save(self, files, src_path=''):
        gauth = self.authenticate()
        drive = GoogleDrive(gauth)
        files = [files] if not isinstance(files, (list, tuple)) else files
        for file in files:
            dir = drive.ListFile({'q': "title = '{}'".format(file)}).GetList()
            f = dir[0] if len(dir) > 0 else drive.CreateFile()
            f.SetContentFile(src_path + file)
            f.Upload()
            print("Put {}".format(file))
