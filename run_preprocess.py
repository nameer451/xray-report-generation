import argparse
from data_process import get_cxr_paths_list, img_to_hdf5, write_report_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_out_path', type=str, required=True)
    parser.add_argument('--cxr_out_path', type=str, default='data/cxr.h5')
    parser.add_argument('--mimic_impressions_path', type=str, default='data/mimic_impressions.csv')
    parser.add_argument('--base_dir', type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. Load image paths
    cxr_paths = get_cxr_paths_list(args.csv_out_path, base_dir=args.base_dir)

    # 2. Save HDF5 of images
    img_to_hdf5(cxr_paths, args.cxr_out_path)

    # 3. Save impressions CSV
    write_report_csv(cxr_paths, args.csv_out_path, args.mimic_impressions_path)

    print("Preprocessing completed.")

        


