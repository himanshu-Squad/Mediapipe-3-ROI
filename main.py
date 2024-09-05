import argparse
import os
from vitallens import VitalLens, Method
import numpy as np
import csv
import pandas as pd
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run VitalLens for vital sign estimation from video.")
    parser.add_argument('--method', type=str, default='VITALLENS', choices=['VITALLENS', 'POS', 'CHROM', 'G'],
                        help='Inference method to use.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--api_key', type=str, required=False, help='API key for VitalLens API.')
    parser.add_argument('--detect_faces', type=bool, default=True, help='Whether to detect faces in the video.')
    parser.add_argument('--estimate_running_vitals', type=bool, default=True, help='Whether to estimate running vitals.')
    parser.add_argument('--export_to_json', type=bool, default=True, help='Whether to export results to a JSON file.')
    parser.add_argument('--export_dir', type=str, default='.', help='Directory to export JSON files.')
    parser.add_argument('--csv_filename', type=str, default='vital_data.csv', help='Filename for the CSV export.')

    return parser.parse_args()

def save_to_csv(rgb_ds: np.ndarray, ppg_waveform: np.ndarray, filename: str):
    """
    Save the rgb_ds and PPG waveform data to a CSV file.

    Args:
        rgb_ds: The downsampled RGB data. Shape (n_frames_ds, 3)
        ppg_waveform: The PPG waveform data. Shape (n_frames,)
        filename: The name of the CSV file to save the data to.
    """
    # Ensure the data lengths match
    assert rgb_ds.shape[0] == ppg_waveform.shape[0], "Data length mismatch between rgb_ds and PPG waveform"

    # Create a DataFrame
    df = pd.DataFrame({
        'R': rgb_ds[:, 0],
        'G': rgb_ds[:, 1],
        'B': rgb_ds[:, 2],
        'PPG_Waveform': ppg_waveform
    })

    # Save to CSV
    df.to_csv(filename, index=False)

def main():
    args = parse_arguments()

    # Ensure the video file exists
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file {args.video_path} does not exist.")

    # Set up the VitalLens instance
    method = getattr(Method, args.method)
    vl = VitalLens(
        method=method,
        api_key=args.api_key,
        detect_faces=args.detect_faces,
        estimate_running_vitals=args.estimate_running_vitals,
        export_to_json=args.export_to_json,
        export_dir=args.export_dir
    )

    # Run the estimation
    result = vl(args.video_path)

    # Print the results
    for face_result in result:
        print("Face coordinates:", face_result['face']['coordinates'])
        print("Heart rate:", face_result['vital_signs']['heart_rate'])
        #print("Respiratory rate:", face_result['vital_signs']['respiratory_rate'])
        print("PPG waveform:", face_result['vital_signs']['ppg_waveform'])
        #print("Respiratory waveform:", face_result['vital_signs']['respiratory_waveform'])
        print("Message:", face_result['message'])
        
        # Retrieve the RGB and PPG waveform data
        if 'ppg_waveform' in face_result['vital_signs']:
            ppg_waveform = face_result['vital_signs']['ppg_waveform']['data']
            if 'rgb_ds' in face_result:
                rgb_ds = face_result['rgb_ds']
                # Save the data to a CSV file
                save_to_csv(np.array(rgb_ds), np.array(ppg_waveform), args.csv_filename)
            else:
                #save_to_csv(np.array(ppg_waveform), args.csv_filename)
                print("rgb_ds not found in the vital signs data.")
        else:
            print("PPG waveform not found in the vital signs data.")
            

if __name__ == "__main__":
    main()
