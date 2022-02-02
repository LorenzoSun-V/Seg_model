import os
import glob


def unzip_files(root_path, dst_path):
    file_dirs = glob.glob(os.path.join(root_path, "*.tar.gz"))
    print(len(file_dirs))
    for file_dir in file_dirs:
        os_cmd = f"tar -xvf {file_dir} -C {dst_path}"
        os.system(os_cmd)
        print(os_cmd)


if __name__ == "__main__":
    # Biome
    # zip_file_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_raw"
    # unzip_file_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_unzip"

    # Irish
    zip_file_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Irish_raw"
    unzip_file_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Irish_unzip"
    unzip_files(zip_file_path, unzip_file_path)