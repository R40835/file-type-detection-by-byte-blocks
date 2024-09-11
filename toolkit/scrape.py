
import os
import shutil
import zipfile
import requests


class FileFormatError(Exception):
    def __init__(self, message: str="The file name does not end with \".zip\"."):
        """
        Exception raised for errors in the file format.
        """
        self.message = message 
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class Govdocs1Api:
    def __init__(self, start_sample: int=-1, end_sample: int=-1, all_samples: bool=False):
        """
        Provides a convinient way to download GovDocs1 dataset. You can either provide 
        a range to denote the zip files you're interesting in, or download all the 
        1000 samples at once.

        Args:
            start_sample (int): The first sample.
            end_sample (int): The last excluded sample (1000 max)
            all_samples (bool): True to download all dataset samples.
        """
        self.working_dir = os.getcwd()
        self.data_dir = "govdocs1"
        self.url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/"
        self.all_samples = all_samples
        if self.all_samples == True:
            self.num_samples = 1000
        else:
            self.start_sample = start_sample
            self.end_sample = end_sample
            self.num_samples = end_sample - start_sample

    def download_dataset(self) -> None:
        """
        Downloads the govdocs1 dataset while updating the user on the progress.
        """
        print("Starting the download. The process might take some time.")
        os.makedirs(self.data_dir, exist_ok=True)
        if self.all_samples:
            for sample_no in range(self.num_samples):
                self._download_govdocs1(sample_no)
        else:
            for sample_no in range(self.start_sample, self.end_sample):
                self._download_govdocs1(sample_no)
        print("Dataset download completed successfully.")

    def _download_govdocs1(self, sample_no: int) -> None:
        """
        Helper method. It downloads a given zip file from govdocs1's server.

        Args:
            sample_no (int): The zip file index number. 
        """
        if len(str(sample_no)) == 1:
            file_name = f"00{sample_no}.zip"
        elif len(str(sample_no)) == 2:
            file_name = f"0{sample_no}.zip"
        elif len(str(sample_no)) == 3:
            file_name = f"{sample_no}.zip"

        url = self.url + file_name
        response = requests.get(url)

        if response.status_code == 200:
            with open(f"{self.data_dir}/{file_name}", "wb") as file:
                file.write(response.content)
            self._exctract_files(file_name)
            self._track_progress(file_name, sample_no)
        else:
            print(f"Failed to download {file_name}. Status code: {response.status_code}")

    def _exctract_files(self, file_name: str) -> None:
        """
        Helper method. Creates a temporary folder to extract files from a zip file 
            then move extracted files to the main dataset directory and deletes
            both the temporary folder and zip file.

        Args:
            file_name (str): The zip file name.
        """
        zip_file_path = os.path.join(self.working_dir, f"{self.data_dir}/{file_name}")
        extraction_dir = os.path.join(self.working_dir, self.data_dir)
        temp_dir = os.path.join(self.working_dir, 'temp')

        os.makedirs(temp_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            if file_name.endswith('.zip'):
                folder_extracted = file_name[:-4] + '/'
            else:
                raise FileFormatError("The file name does not end with \".zip\".")
            
            # move files from the extracted folder to the target directory
            for root, dirs, files in os.walk(os.path.join(temp_dir, folder_extracted)):
                for file in files:
                    file_path = os.path.join(root, file)
                    target_path = os.path.join(extraction_dir, file)
                    shutil.move(file_path, target_path)
            
            shutil.rmtree(temp_dir)
            os.remove(zip_file_path)

        except zipfile.BadZipFile:
            print("The ZIP file is corrupted or not a valid ZIP file.")
        except FileNotFoundError:
            print("The specified ZIP file was not found.")
        except PermissionError:
            print("Permission denied: unable to write to the extraction directory.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _track_progress(self, file_name: str, sample_no: int) -> None:
        """
        Helper method. Tracks the progress of the dataset download.

        Args:
            file_name (str): The downloaded zip file name.
            sample_no (int): The zip file sample number.
        """
        if self.all_samples:
            percent_progress = (sample_no / self.num_samples) * 100
            print(f"{file_name} downloaded. {sample_no}/{self.num_samples} ({percent_progress:.2f}%)")
        else:
            percent_progress = (sample_no - self.start_sample + 1) / self.num_samples * 100
            print(f"{file_name} downloaded. {sample_no - self.start_sample + 1}/{self.num_samples} ({percent_progress:.2f}%)")