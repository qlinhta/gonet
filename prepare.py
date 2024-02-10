import os
import requests
import zipfile
from io import BytesIO
import subprocess  # For executing git commands


def download_and_extract_zip(url, extract_to='./data'):
    print(f"Downloading zip file from {url}")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        print(f'Extracting files to {extract_to}')
        zip_file.extractall(extract_to)
    print("Extraction complete!")


def move_file(source, destination):
    if not os.path.exists(os.path.dirname(destination)):
        print(f"Creating directory for {destination}...")
        os.makedirs(os.path.dirname(destination))

    print(f"Moving {source} to {destination}...")
    os.rename(source, destination)
    print(f"File '{os.path.basename(destination)}' has been moved to '{destination}'.")


def setup_git_config(email, name):
    print("Configuring git user email and name...")
    subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name', name], check=True)
    print("Git configuration completed.")


file_url = "https://www.lamsade.dauphine.fr/~cazenave/project2022.zip"
source_file_path = "./data/games.data"
destination_directory = "./"
destination_file_path = os.path.join(destination_directory, "games.data")

download_and_extract_zip(file_url)
move_file(source_file_path, destination_file_path)

email = "qlinhta@outlook.com"
name = "Quyen Linh TA"
setup_git_config(email, name)
print("Git configuration completed.")
