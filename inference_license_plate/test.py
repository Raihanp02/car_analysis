from fast_plate_ocr import LicensePlateRecognizer

# This downloads the .onnx + plate_config.yaml to a local cache dir
m = LicensePlateRecognizer('cct-xs-v2-global-model')

from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
for repo in cache_info.repos:
    print(repo.repo_id, "-", repo.repo_type)
    for revision in repo.revisions:
        for f in revision.files:
            print("  ", f.file_path)