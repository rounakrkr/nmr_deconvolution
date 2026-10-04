NMR PROJECT DATASET

COMPONENTS
- 30 synthetic 1H NMR component spectra.
- Each component has 16,384 points from 10 to 0 ppm.
- CSV and NPY contain the numerical data.
- PNG is generated directly from the same numerical data.
- Component peak positions are based on chemically plausible/original NMR peak locations.
- Intensities are synthetic arbitrary units, not claimed experimental measurements.

MIXTURES
- 1,000 datasets.
- 20 mixtures per dataset.
- 5 components per mixture.
- 20,000 mixture spectra total.
- No 5-component combination repeats between datasets.
- Concentration fractions differ between mixtures within a dataset.
- Each mixture has 16,384 intensity values.
- Mixtures are linear concentration-weighted sums of the component spectra.
- No artificial random noise or baseline distortion is added.
- Each dataset_XXXX.npz contains:
    spectra: shape (20, 16384), float16
    ppm: shape (16384,)
- mixture_metadata.csv gives the five component identities and fractions for every mixture.

This is a synthetic ML/project dataset. It is not a collection of experimentally measured mixtures.
