import requests
import zipfile
import os
import shutil
from tqdm import tqdm

def download():
    url = 'https://huggingface.co/Ailyth/Text_to_Speech_MODELS/resolve/main/tts_models.zip?download=true'
    output = 'temp.zip'
    current_dir = os.path.dirname(os.path.abspath(__file__))

    info_py_path = os.path.join(current_dir, 'info.py')
    models_dir = os.path.join(current_dir, 'MODELS')
    if os.path.exists(info_py_path) and os.path.exists(models_dir):
        print('Files already exist, skipping download.')
        return

    print('✨ Downloading models')
    response = requests.head(url)
    total_size = int(response.headers.get('content-length', 0))

    block_size = 1024  
    progress = tqdm(total=total_size, unit='iB', unit_scale=True)
    with requests.get(url, stream=True) as r:
        with open(output, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    progress.close()

    with zipfile.ZipFile(output, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        print('Installing')
        extract_progress = tqdm(total=total_files, unit='file', leave=False)
        for file_info in zip_ref.infolist():
            try:
                zip_ref.extract(file_info, path=current_dir)
            except zipfile.error as e:
                print(f"Error extracting {file_info.filename}: {e}")
            extract_progress.update()
        extract_progress.close()

    os.remove(output)

if __name__ == '__main__':
    download()