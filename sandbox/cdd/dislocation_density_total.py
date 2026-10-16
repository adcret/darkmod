import os
import numpy as np


def main():
    # Get input folder and output file
    folder = "/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Adam/CDD/Microstructures/"
    output_file = "dislocation_density_total.npy"
    
    # Find all .npy files in the folder
    files = [f for f in os.listdir(folder) if f.startswith('dislocation_density_') and f.endswith('.npy')]
    files.sort()
    
    
    if not files:
        print("No .npy files found in folder.")
        return
    
    # Process each file
    means = []
    for filename in files:
        filepath = os.path.join(folder, filename)
        try:
            data = np.load(filepath)
            mean_val = np.mean(data)
            means.append(mean_val)
            print(f"Processed {filename}: mean = {mean_val:.6e}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if not means:
        print("No valid results to save.")
        return
    
    # Save results
    results = np.array(means)
    np.save(output_file, results)
    print(f"\nSaved {len(means)} means to {output_file}")


if __name__ == "__main__":
    main()

