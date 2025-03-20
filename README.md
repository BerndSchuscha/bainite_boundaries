# Bainite Boundaries

## Overview
Bainite Boundaries is a materials science project focused on analyzing and modeling boundaries of the bainite composition space.


## Installation

To set up the project environment:

1. Clone the repository:
   ```sh
   git clone https://github.com/BerndSchuscha/bainite_boundaries.git
   ```
2. Navigate to the project directory:
   ```sh
   cd bainite_boundaries
   ```
3. Create and activate a Conda environment:
   ```sh
   conda env create -f environment.yml
   conda activate bainite_boundaries
   ```
4. Install the package in the folder:
   ```sh
   pip install .
   ```

## Usage

### Running the Main Calculation
The primary script for calculations is located at:
   ```sh
   bainite_boundaries/BC_calculation
   ```
The evulation points are in
   ```
   data/final_samples.txt
   ```

### Evaluate results
The result is evaluted in the Jupyter notebook:
   ```sh
   jupyter notebook bainite_boundaries/visualization/make_final_plot.ipynb
   ```


## Contact

For questions, suggestions, or collaboration inquiries, please contact **Bernd Schuscha**.



## Authors:
   - Bernd Schuscha (bernd.schuscha@mcl.at)
