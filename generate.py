import argparse
import requests
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import elevation
import rasterio

def get_elevation_data(start_lat, start_lon, end_lat, end_lon):
    output_file = 'elevation.tif'
    print(output_file)
    elevation.clip(bounds=(start_lon, start_lat, end_lon, end_lat), output=output_file)
    with rasterio.open(output_file) as src:
        elevation_data = src.read(1)
    return elevation_data

def extract_middle_profile(elevation_data):
    height, width = elevation_data.shape
    if height > width:
        middle_index = height // 2
        profile = elevation_data[:, middle_index]
    else:
        middle_index = width // 2
        profile = elevation_data[middle_index, :]
    return profile

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def map_to_frequency(data, min_freq, max_freq):
    return data * (max_freq - min_freq) + min_freq

def create_sound_wave(frequencies, duration):
    burst_duration = duration / len(frequencies)
    sound = AudioSegment.empty()
    for freq in frequencies:
        burst = Sine(freq).to_audio_segment(duration=burst_duration * 1000)
        sound += burst
    return sound

def export_sound_wave(sound, filename):
    sound.export(filename, format='wav')

def main(start_lat, start_lon, end_lat, end_lon, duration, output_path):
    elevation_data = get_elevation_data(start_lat, start_lon, end_lat, end_lon)
    profile = extract_middle_profile(elevation_data)
    normalized_profile = normalize_data(profile)
    frequency_profile = map_to_frequency(normalized_profile, 500, 5000)
    sound_wave = create_sound_wave(frequency_profile, duration)
    export_sound_wave(sound_wave, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a sound wave from elevation data.')
    parser.add_argument('start_lat', type=float, help='Starting latitude of the cross-section')
    parser.add_argument('start_lon', type=float, help='Starting longitude of the cross-section')
    parser.add_argument('end_lat', type=float, help='Ending latitude of the cross-section')
    parser.add_argument('end_lon', type=float, help='Ending longitude of the cross-section')
    parser.add_argument('duration', type=int, help='Duration of the sound wave in seconds')
    parser.add_argument('output_path', type=str, help='Path to output WAV file')

    args = parser.parse_args()
    main(args.start_lat, args.start_lon, args.end_lat, args.end_lon, args.duration, args.output_path)
