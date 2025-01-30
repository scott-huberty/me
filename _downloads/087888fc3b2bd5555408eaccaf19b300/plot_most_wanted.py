"""
Use ICA to separate a 100 gecs song into its components
=======================================================

We'll use the "Most Wanted Person in the United States" stems from the 10,000 gecs
stems pack to demonstrate how to use ICA to separate the components of a song.
"""

from pathlib import Path

import IPython.display as ipd

import numpy as np

from sklearn.decomposition import FastICA

from scipy.io import wavfile

# %%
# Load the mixed audio
# ---------------------

# %%
SOURCES_DIR = (
    Path(".").expanduser().resolve().parent /
    "assets" /
    "mixed_audio"
)
assert SOURCES_DIR.exists()

# %%
sfreq, drums_mic = wavfile.read(SOURCES_DIR / "drums.wav")
bass_mic = wavfile.read(SOURCES_DIR / "bass.wav")[1]
guitars_mic = wavfile.read(SOURCES_DIR / "guitars.wav")[1]
fx_mic = wavfile.read(SOURCES_DIR / "fx.wav")[1]
vocals_mic = wavfile.read(SOURCES_DIR / "vocals.wav")[1]
mix = drums_mic + bass_mic + guitars_mic + fx_mic + vocals_mic
mix = mix / mix.max()

# %%
# Here is the mixed audio, and the bass, fx, and vocals isolated (for reference)
# -------------------------------------------------------------------------------

# %%
ipd.Audio(mix, rate=sfreq)

# %%
ipd.Audio(bass_mic, rate=sfreq)

# %%
ipd.Audio(vocals_mic, rate=sfreq)

# %%
ipd.Audio(fx_mic, rate=sfreq)

# %%
# Separate the components with ICA
# --------------------------------
# We'll stack our "observations" (the microphone recordings) into a matrix.
# Each column will be a different microphone, and each row will be a different
# time point. We'll then use ICA to separate the components. By components, we
# mean the original sources that were mixed together to create the microphone
# recordings (drums, bass, guitars, fx, and vocals).

# %%
microphones = np.vstack([drums_mic, bass_mic, guitars_mic, fx_mic, vocals_mic]).T
ica = FastICA()
components = ica.fit_transform(microphones)
# Unpack the components
ic_1, ic_2, ic_3, ic_4, ic_5 = components.T

# %%
# Here are the separated components
# ---------------------------------

# %%
ipd.Audio(ic_1, rate=sfreq)

# %%
ipd.Audio(ic_2, rate=sfreq)

# %%
ipd.Audio(ic_3, rate=sfreq)

# %%
# Not too bad!
# ------------
# The ICA algorithm was able to separate the components pretty well.
# The separated components are not exactly the same as the original sources, but
# they are pretty close. The ICA algorithm is able to separate the components
# because the sources are statistically independent. This is a pretty cool
# demonstration of the power of ICA!