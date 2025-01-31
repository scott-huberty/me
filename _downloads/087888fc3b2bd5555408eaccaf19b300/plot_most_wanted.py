"""
Use ICA to separate the instruments in a 100 Gecs song.
=======================================================

.. image:: https://i.kym-cdn.com/entries/icons/original/000/018/666/How_Do_You_Do_Fellow_Kids_meme_banner_image.jpg


ICA is a wonderful algorithm. It can be used tease apart different "sources" in a
signal.

But there's a catch. You need to have multiple "observations" of this signal.
The canonical example would be an orchestra performance recorded with multiple
microphones (a few in each section). Each microphone is one "observation". Since
each microphone picks up a blend of all the instruments, you can use ICA to separate
the individual instruments (so you'd have the violins isolated.. 0r the tuba.. etc.).

And no, you probably can't use ICA to separate the vocals from a song you downloaded
from the internet.

Since a an orchestra performance might bore you (and because I don't have such
a recording handy), let's use a different example. Pretend you are in the studio with
100 gecs. You set up 4 microphones in the room, each recording the drums, bass, an 
effects track (FX), and vocals. The individual microphones also pick
up some of the other instruments, but that's okay. You can use ICA to separate the
components!
"""

from functools import partial
from pathlib import Path

import IPython.display as ipd
import numpy as np
import pooch
from scipy.io import wavfile
from sklearn.decomposition import FastICA

# %%
# Define some helper functions
# ----------------------------
# (You can skip this section if you're not interested in the details)


# %%
def load_audio(wav_path):
    """Load a wav file from disk."""
    return wavfile.read(wav_path)


def convert_to_mono(wav_array):
    """Convert stereo audio to mono by averaging the channels."""
    return np.mean(wav_array, axis=1)


def normalize_audio(wav_array):
    """Normalize the decibel range to -1 to 1."""
    return wav_array / np.max(np.abs(wav_array))


def process_audio(wav_path):
    """Load a wav file, convert stereo to mono, and normalize decibel range."""
    sfreq, wav_array = load_audio(wav_path)
    if len(wav_array.shape) > 1:
        wav_array = convert_to_mono(wav_array)
    return sfreq, normalize_audio(wav_array)


def mix_stems(*wavs, mix_matrix):
    """Blend the individual stems together using a mixing matrix."""
    return np.dot(mix_matrix, np.array(wavs, dtype=float))


# %%
# Load the mixed audio
# ---------------------
# We'll define a data fetcher to download the stems pack from the 10,000 gecs album.
# Please note that this will download a 1.2 GB file to your machine. Please be patient!

# %%
print("Please be patient, this may take a while...")
# We will ignore the guitars stem because it is mostly silent
want_stems = ["Drums.wav", "Bass.wav", "Vocals.wav", "FX.wav"]
members = [
    f"10,000 gecs Stems/The Most Wanted Person in the United States/{stem}"
    for stem in want_stems
]

unpack = pooch.Unzip(
    extract_dir=".", # Relative to the path where the zip file is downloaded
    members=members,
)

stem_fpaths = pooch.retrieve(
    url="https://www.100gecs.com/uploads/10000gecsstems.zip",
    known_hash="sha256:65d2f8dc5cf61a6cd2ac722c2c3bef465b76ca50f5d0363425acdbc5b100e754",
    progressbar=True,
    path=Path.home() / "100gecs",
    processor=unpack,
)
stems_dir = Path(stem_fpaths[0]).parent

# %%
# Load the stems
# --------------
# We'll load the stems and process them by converting stereo to mono and normalizing
# the decibel range.

# %%
sfreq, drums = process_audio(stems_dir / "Drums.wav")
# For memory purposes, let's cut the recording in half
n_samples = drums.shape[0]
crop = n_samples // 2

drums = drums[:crop]
bass = process_audio(stems_dir / "Bass.wav")[1][:crop]
fx = process_audio(stems_dir / "FX.wav")[1][:crop]
vocals = process_audio(stems_dir / "Vocals.wav")[1][:crop]

mix_matrix = np.array([0.50, 0.20, 0.15, 0.15])
mix_func = partial(mix_stems, mix_matrix=mix_matrix)

drums = mix_func(drums, bass, fx, vocals)
bass = mix_func(bass, fx, vocals, drums)
fx = mix_func(fx, vocals, drums, bass)
vocals = mix_func(vocals, drums, bass, fx)

# %%
# Here is one (blended) stem for reference
# ----------------------------------------

# %%
ipd.Audio(fx, rate=sfreq)

# %%
# Separate the components with ICA
# --------------------------------
# We'll stack our "observations" (the microphone recordings) into a matrix.
# Each column will be a different microphone, and each row will be a different
# time point. We'll then use ICA to separate the components. By components, we
# mean the original sources that were mixed together to create the microphone
# recordings (drums, bass, guitars, fx, and vocals).

# %%
microphones = np.vstack([drums, bass, fx, vocals]).T
ica = FastICA(random_state=42)
components = ica.fit_transform(microphones)
# Unpack the components
ic_1, ic_2, ic_3, ic_4 = components.T

# %%
# Here are the separated components
# ---------------------------------

# %%
ipd.Audio(normalize_audio(ic_1), rate=sfreq)

# %%
ipd.Audio(normalize_audio(ic_2), rate=sfreq)

# %%
ipd.Audio(normalize_audio(ic_3), rate=sfreq)

# %%
# Not too bad!
# ------------
# The ICA algorithm was able to separate the components pretty well.
# The separated components are not exactly the same as the original sources, but
# they are pretty close. The ICA algorithm is able to separate the components
# because the sources are statistically independent. This is a pretty cool
# demonstration of the power of ICA!
