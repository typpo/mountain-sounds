# Mountain Sounds

Mountain Sounds is a Python script that generates a sound wave from elevation data. It can take either a GPX file or start/end coordinates as input, and it outputs a sound wave file, an image of the frequency profile, an image of the elevation profile, and a video that combines the sound wave and elevation profile.

Here's an example video output:

https://github.com/typpo/mountain-sounds/assets/310310/49fef4ae-c5e9-42b1-b0df-fe70fff66279

## Usage

There are two ways to use this script: with a GPX file or with start/end coordinates.

### GPX File

To use a GPX file, run the script with the `--gpx` argument followed by the path to the GPX file. For example:

```bash
python generate.py --gpx path/to/file.gpx --duration 10 --output_prefix output
```

This command will generate a sound wave from the elevation data in the GPX file. The sound wave will last for 10 seconds and the output files will be prefixed with "output".

### Start/End Coordinates

To use start/end coordinates, run the script with the `--start_lat`, `--start_lon`, `--end_lat`, and `--end_lon` arguments followed by the respective coordinates. For example:

```bash
python generate.py --start_lat 37.7749 --start_lon -122.4194 --end_lat 34.0522 --end_lon -118.2437 --duration 10 --output_prefix output
```

This command will generate a sound wave from the elevation data between the coordinates (37.7749, -122.4194) and (34.0522, -118.2437). The sound wave will last for 10 seconds and the output files will be prefixed with "output".

## Output

The script outputs four files:

- A .wav file containing the sound wave
- A .png image of the frequency profile
- A .png image of the elevation profile
- A .mp4 video that combines the sound wave and elevation profile

The prefix for these files is specified by the `--output_prefix` argument.

## Dependencies

This script uses several Python libraries, including gpxpy, elevation, matplotlib, numpy, pydub, rasterio, geopy, scipy, and moviepy. To get started, run:

```bash
poetry install
```

Or they can be installed with pip:

```bash
pip install gpxpy elevation matplotlib numpy pydub rasterio geopy scipy moviepy
```
