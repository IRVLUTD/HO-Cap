import zipfile
import requests
from hocap.utils import *


def download_box_file(box_link, output_file):
    output_path = Path(output_file)
    file_name = output_file.name

    resume_header = {}
    downloaded_size = 0

    with requests.get(box_link, headers=resume_header, stream=True) as response:
        # Check if the request was successful
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
        else:
            print(f"Failed to retrieve file info. Status code: {response.status_code}")
            return

    if output_path.exists():
        downloaded_size = output_path.stat().st_size
        # Check if there's a partial download and get its size
        resume_header = {"Range": f"bytes={downloaded_size}-"}

    # Check if the file is already fully downloaded
    if downloaded_size == total_size:
        tqdm.write(f"  ** {file_name} is already downloaded.")
        return

    # Send a GET request with the range header if needed
    with requests.get(box_link, headers=resume_header, stream=True) as response:
        # Check if the request was successful
        if response.status_code in [200, 206]:
            # Initialize tqdm progress bar
            with tqdm(
                total=total_size,
                initial=downloaded_size,
                unit="B",
                unit_scale=True,
                ncols=80,
            ) as pbar:
                # Download the file in chunks
                with output_path.open("ab") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):  # 1 MB chunks
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


def unzip_file(zip_file, output_dir):
    zip_file = Path(zip_file)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def args_parser():
    parser = argparse.ArgumentParser(description="Download dataset files")
    parser.add_argument(
        "--subject_id",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        help="The subject number to download",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    dataset_files = read_data_from_json(PROJ_ROOT / "config/hocap_recordings.json")

    tqdm.write(f"- Downloading 'models.zip'...")
    download_box_file(dataset_files["models.zip"], PROJ_ROOT / "data/models.zip")

    tqdm.write(f"- Downloading 'calibration.zip'...")
    download_box_file(
        dataset_files["calibration.zip"], PROJ_ROOT / "data/calibration.zip"
    )

    if args.subject_id == "all":
        for i in range(1, 10):
            for file_name, file_link in dataset_files[f"subject_{i}"].items():
                if len(file_link) == 0:
                    continue
                tqdm.write(f"- Downloading 'subject_{i}/{file_name}'...")
                download_box_file(
                    file_link, PROJ_ROOT / f"data/subject_{i}/{file_name}"
                )
    else:
        for file_name, file_link in dataset_files[f"subject_{args.subject_id}"].items():
            if len(file_link) == 0:
                continue
            tqdm.write(f"- Downloading 'subject_{args.subject_id}/{file_name}'...")
            download_box_file(
                file_link, PROJ_ROOT / f"data/subject_{args.subject_id}/{file_name}"
            )

    # Extract the downloaded zip files
    zip_files = list(PROJ_ROOT.glob("data/*.zip"))
    tqdm.write(f"- Extracting downloaded zip files...")
    for zip_file in zip_files:
        tqdm.write(f"  ** Extracting '{zip_file.name}'...")
        unzip_file(zip_file, zip_file.parent)
