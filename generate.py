import argparse
import math
import os
import tempfile

import gpxpy
import elevation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from geopy.distance import great_circle, geodesic
from geopy.point import Point
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from pydub.generators import Sine
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter


def get_elevation_data(start_lat, start_lon, end_lat, end_lon):
    padding = 0.01  # Padding for bounding box, in degrees
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
        output_file = temp_file.name

        min_lat = min(start_lat, end_lat) - padding
        max_lat = max(start_lat, end_lat) + padding
        min_lon = min(start_lon, end_lon) - padding
        max_lon = max(start_lon, end_lon) + padding
        elevation.clip(bounds=(min_lon, min_lat, max_lon, max_lat), output=output_file)
        with rasterio.open(output_file) as src:
            elevation_data = src.read(1)

        # Clean up the temporary file
        os.remove(temp_file.name)
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


def extract_elevation_profile_latlng(
    elevation_data, start_lat, start_lon, end_lat, end_lon, num_points=1000
):
    start = Point(start_lat, start_lon)
    end = Point(end_lat, end_lon)

    distance = great_circle(start, end).miles
    step = distance / num_points

    latitudes = np.linspace(start_lat, end_lat, elevation_data.shape[0])
    longitudes = np.linspace(start_lon, end_lon, elevation_data.shape[1])
    interpolator = interp2d(longitudes, latitudes, elevation_data, kind="cubic")

    dlon = math.radians(end.longitude - start.longitude)
    lat1 = math.radians(start.latitude)
    lat2 = math.radians(end.latitude)
    bearing = math.atan2(
        math.sin(dlon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon),
    )
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    profile = []
    distances = []
    for i in range(num_points):
        point = great_circle(miles=i * step).destination(point=start, bearing=bearing)
        elevation = interpolator(point.longitude, point.latitude)
        profile.append(elevation[0])
        distances.append(i * step)

    return np.array(profile), np.array(distances)


def extract_elevation_profile_from_gpx(gpx_path, num_points=1000):
    with open(gpx_path, "r") as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    points = gpx.tracks[0].segments[0].points
    elevations = [point.elevation for point in points]
    distances = [0]
    for i in range(1, len(points)):
        distances.append(
            distances[i - 1]
            + geodesic(
                (points[i - 1].latitude, points[i - 1].longitude),
                (points[i].latitude, points[i].longitude),
            ).miles
        )
    # Interpolate to get the desired number of points
    x = np.linspace(0, len(elevations) - 1, num_points)
    y = np.interp(x, np.arange(len(elevations)), elevations)
    d = np.interp(x, np.arange(len(distances)), distances)
    return y, d


def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def map_to_frequency(data, min_freq, max_freq):
    power = 1.0  # Change this value to adjust the amount of contrast
    return (data**power) * (max_freq - min_freq) + min_freq


def smooth(data, sigma=5):
    return gaussian_filter(data, sigma)


class FrequencySweep(Sine):
    def __init__(self, start_frequency, end_frequency, duration, **kwargs):
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.duration = duration
        self.amplitude = kwargs.get("amplitude", 1)
        super().__init__(start_frequency, **kwargs)

    def generate(self):
        for i, sample in enumerate(super().generate()):
            time = i / self.sample_rate
            frequency = (
                self.start_frequency
                + (self.end_frequency - self.start_frequency) * time / self.duration
            )
            yield self.amplitude * math.sin(2 * math.pi * frequency * time)


def create_sound_wave(frequencies, duration):
    burst_duration = duration / len(frequencies)
    sound = AudioSegment.empty()
    for i in range(len(frequencies) - 1):
        start_freq = frequencies[i]
        end_freq = frequencies[i + 1]
        burst = FrequencySweep(start_freq, end_freq, burst_duration).to_audio_segment(
            duration=burst_duration * 1000
        )
        fade_duration = min(
            50, burst_duration * 500
        )  # Ensure fade duration is not longer than burst duration
        burst = burst.fade_in(fade_duration).fade_out(fade_duration)
        sound += burst
    return sound


def export_sound_wave(sound, filename):
    sound.export(filename, format="wav")


def export_elevation_profile(profile, distances, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, profile)
    plt.title("Elevation Profile")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Elevation")
    plt.grid(True)
    plt.savefig(filename)


def export_sound_wave_image(frequency_profile, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(frequency_profile)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig(filename)


def export_video(
    elevation_profile,
    distances,
    audio_path,
    video_path,
    duration,
    title=None,
    framerate=60,
):
    assert len(elevation_profile) == len(
        distances
    ), "There must be a distance value for each elevation value"

    total_frames = duration * framerate
    x = np.linspace(0, len(elevation_profile) - 1, total_frames)
    interpolated_elevation_profile = np.interp(
        x, np.arange(len(elevation_profile)), elevation_profile
    )
    interpolated_distances = np.interp(x, np.arange(len(distances)), distances)

    fig, ax = plt.subplots()
    if title is not None:
        plt.title(title)
    (line,) = ax.plot(
        interpolated_distances, interpolated_elevation_profile * 3.281
    )  # Convert from meters to feet
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Elevation (ft)")

    def update(num, interpolated_elevation_profile, line):
        line.set_data(
            interpolated_distances[: num + 1],
            interpolated_elevation_profile[: num + 1] * 3.281,
        )  # Update both x and y data and convert from meters to feet
        print(f"Rendering frame {num+1} of {total_frames}")
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        total_frames,
        fargs=[interpolated_elevation_profile, line],
        interval=1000 / framerate,
    )

    # Save the animation as a video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        ani.save(temp_video.name, writer="ffmpeg")

        # Add the audio track to the video
        video = VideoFileClip(temp_video.name)
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
        video.write_videofile(video_path)

        # Clean up the temporary video file
        os.remove(temp_video.name)


def main(start_lat, start_lon, end_lat, end_lon, duration, output_prefix, gpx):
    if gpx is not None:
        profile, distances = extract_elevation_profile_from_gpx(gpx, duration * 100)
    elif (
        start_lat is not None
        and start_lon is not None
        and end_lat is not None
        and end_lon is not None
    ):
        elevation_data = get_elevation_data(start_lat, start_lon, end_lat, end_lon)
        profile, distances = extract_elevation_profile_latlng(
            elevation_data, start_lat, start_lon, end_lat, end_lon, duration * 100
        )
    else:
        raise Exception("Either a GPX file or start/end coordinates must be provided.")

    normalized_profile = normalize_data(profile)
    frequency_profile = map_to_frequency(normalized_profile, 10, 500)
    smoothed_frequency_profile = smooth(frequency_profile)
    sound_wave = create_sound_wave(smoothed_frequency_profile, duration)
    export_sound_wave(sound_wave, f"{output_prefix}_sound.wav")
    export_sound_wave_image(
        smoothed_frequency_profile, f"{output_prefix}_frequency_profile.png"
    )
    export_elevation_profile(
        profile, distances, f"{output_prefix}_elevation_profile.png"
    )
    export_video(
        profile,
        distances,
        f"{output_prefix}_sound.wav",
        f"{output_prefix}_video.mp4",
        duration,
        title=args.title,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a sound wave from elevation data."
    )
    parser.add_argument(
        "--start_lat", type=float, help="Starting latitude of the cross-section"
    )
    parser.add_argument(
        "--start_lon", type=float, help="Starting longitude of the cross-section"
    )
    parser.add_argument(
        "--end_lat", type=float, help="Ending latitude of the cross-section"
    )
    parser.add_argument(
        "--end_lon", type=float, help="Ending longitude of the cross-section"
    )
    parser.add_argument("--gpx", type=str, help="Path to GPX file")
    parser.add_argument(
        "--duration", type=int, help="Duration of the sound wave in seconds"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Prefix for the output files",
        default="output",
    )
    parser.add_argument("--title", type=str, help="Title for the video")

    args = parser.parse_args()
    main(
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        end_lat=args.end_lat,
        end_lon=args.end_lon,
        duration=args.duration,
        output_prefix=args.output_prefix,
        gpx=args.gpx,
    )
