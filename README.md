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
   ```
   final_samples
   ```

### Generating Final Plots
To generate final evaluation plots, run the Jupyter notebook:
   ```sh
   jupyter notebook bainite_boundaries/visualization/make_final_plot.ipynb
   ```

### Running k-fold plots
1. Ensure the environment is activated:
   ```sh
   conda activate bainite_boundaries
   ```
2. Open the Jupyter notebook:
   ```sh
   jupyter notebook Make_kfold_plots.ipynb
   ```

## Documentation

To build and view documentation:
1. Install dependencies:
   ```sh
   pip install -r docs/requirements.txt
   ```
2. Build documentation:
   ```sh
   mkdocs build
   ```
3. Serve documentation locally:
   ```sh
   mkdocs serve
   ```
4. Open `http://127.0.0.1:8000/` in your browser.

## Testing

To run the test suite:
   ```sh
   pytest
   ```

## Contributing

We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a new feature branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```sh
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```sh
   git push origin feature-branch
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or collaboration inquiries, please contact **Bernd Schuscha**.



## Authors:
   - Bernd Schuscha (bernd.schuscha@mcl.at)