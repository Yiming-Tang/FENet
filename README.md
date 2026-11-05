# Running the code

Run all commands from this directory with Python 3.12.

## Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Simulation

Run a single-change simulation and localization example:

```bash
python -m simulation.run --scenario 3 --innovations IID --window-length 100 --n-train 1000 --n-test 1000 --epochs 20 --batch-size 32 --learning-rate 0.001 --multiple-length 300 --output-dir outputs/simulation
```

Set `--scenario` to `1`, `2`, or `3`, and `--innovations` to `IID` or `FAR`.
Use `--embedding legacy-notebook` for the original notebook embedding; the
default is `paper`. To use the manuscript's stated optimization settings,
set `--epochs 100 --batch-size 64 --learning-rate 0.1`. These options select
settings; they do not resolve differences between the source data generator
and the manuscript. Set `--device cuda` to train on a CUDA GPU.

Run the scalability and misspecification experiments:

```bash
python -m simulation.scalability --output-dir outputs/scalability
python -m simulation.misspecification --output-dir outputs/misspecification
```

For a short simulation check:

```bash
python -m simulation.run --window-length 20 --n-train 40 --n-test 40 --epochs 2 --multiple-length 80 --output-dir outputs/quick
```

Metrics, predictions, settings, and model weights are saved under the selected
output directory. Add `--help` to any command to list its options.

## Real data: Sleep-EDF

Download and preprocess the data, then run the three-model benchmark:

```bash
python -m real_data.download --data-root data/sleep-edf --nights all
python -m real_data.preprocess --data-root data/sleep-edf --modality eog --nights all
python -m real_data.benchmark --input data/sleep-edf/processed/sleep_edf_eog_two_nights_logpsd_T20.pt --split-strategy sample --split-seeds 2026 2027 2028 2029 2030 --model-seeds 101 202 303 --methods fenet-mlp temporal-cnn klcpd --output outputs/sleep-edf
```

Use `--modality both` to prepare both EEG and EOG. For EEG, pass
`data/sleep-edf/processed/sleep_edf_two_nights_logpsd_T20.pt` to `--input`.
Use `--nights first` in both preparation commands for the first-night subset;
its processed filenames omit `_two_nights`.

`--split-strategy sample` selects the original 70/10/20 window split, which
allows subjects and overlapping recording intervals across subsets.
Use `--split-strategy subject` for subject-disjoint 60/20/20 splits.
Add `--quick --device cpu` for a three-epoch check with one split and one model
seed. Use `--methods fenet-mlp` to run only FENet. Legacy processed files without
raw log-PSD require `--allow-clipped-input`; regenerate them with
`real_data.preprocess` to avoid approximating values lost to clipping.

Results are saved under `outputs/sleep-edf/<split-strategy>/`.

## R comparisons

Install the optional R dependencies:

```r
install.packages(c("fda", "remotes"))
remotes::install_github(
  "SonmezOzan/fChange_0.2.0@2f79241a97faf9cd28d09f500b50f45feaae9122",
  upgrade = "never"
)
```

Export the test windows and run FFM and TEV:

```bash
python -m real_data.benchmark --split-strategy sample --split-seeds 2026 --export-only --output outputs/sleep-edf
Rscript simulation/r/benchmark.R --input-dir outputs/sleep-edf/sample/split_seed_2026/r_test --output outputs/sleep-edf/r_metrics.csv
```

Use `--methods TEV` on the R command to run without `fChange`. R metrics,
predictions, and error details are saved beside the selected output file.
