import requests
import zipfile
import os
from io import BytesIO

file_url = "https://www.lamsade.dauphine.fr/~cazenave/project2022.zip"

response = requests.get(file_url)
zip_file = zipfile.ZipFile(BytesIO(response.content))

zip_file.extractall("./data")

source_file_path = "./data/games.data"
destination_directory = "./"

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

destination_file_path = os.path.join(destination_directory, "games.data")
os.rename(source_file_path, destination_file_path)

print(f"File 'games.data' has been copied to '{destination_file_path}'.")
