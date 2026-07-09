
## Prerequisites

- Python 3.8 or higher
- Required Python packages: `numpy`, `pandas`
- Standard Unix utilities: `bash`, `tar`, `gzip`, `du`, `python3`
- Operating system: Linux or WSL2 (Ubuntu 22.04+ recommended)

## Configuration

All parameters are defined in `config.json`. The following fields control the pipeline:

| Field               | Description                                                                 | Example                          |
|---------------------|-----------------------------------------------------------------------------|----------------------------------|
| `archive_name`      | Base name for the output archive (no extension)                             | `"rybak_exp_v1"`                 |
| `output_filename`   | Name of the generated configuration CSV                                     | `"cfg_weights.csv"`              |
| `base_weight`       | Default weight for all non-Renshaw connections                              | `0.5`                            |
| `renshaw_weights`   | Range and step for Renshaw connection weights                               | `{"min":0.0, "max":1.0, "step":0.2}` |
| `meander`           | Pulse parameters (`period`, `duration`, `phase`). `num` defines `linspace` points | `{"min":100, "max":800, "num":5}`    |
| `base_currents`     | Baseline currents `[I1, I2, I3, I4]`                                        | `[0, 0, 0, 0]`                   |
| `noise_percent`     | Noise level applied during simulation                                       | `0.05`                           |

## Usage

1. Ensure all scripts are located in the `src/` directory.
2. Edit `config.json` to match your experiment requirements.
3. Make the pipeline executable and run it:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```
# Run pipeline without cleanup (default behavior)
bash run.sh

# Run pipeline and automatically delete ../data after successful archiving
bash run.sh --cleanup
# or
bash run.sh -c