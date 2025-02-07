"""

Predict hand pose from EMG signals using ML
===========================================

Meta has been doing a lot of work trying to figure out how to replace
keyboards and controllers with hand gestures (when the user is wearing
a wrist band). 

Back when I was a young hopper (AKA a PhD student), I interviewed with
their Reality Labs team. The technical interview at the time was to
take a dataset of EMG signals from a participant, and to predict the
hand pose of the participant (Also, nearly pure python.. I.e. no
scikit-learn, no pytorch. Woof).

Well now they've open-sourced the EMG dataset from that project, so
I am going to save some soul out there some time and show how them
how to do it.
"""

# %%
from collections.abc import KeysView
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pooch
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# %%
# Create a helper function to read the data
# ------------------------------------------------
#

# %%
@dataclass
class Emg2PoseSessionData:
    """A read-only interface to a single emg2pose session file stored in
    HDF5 format.

    ``self.timeseries`` is a `h5py.Dataset` instance with a compound data type
    as in a numpy structured array containing three fields - EMG data from the
    left and right wrists, and their corresponding timestamps.
    The sampling rate of EMG is 2kHz, each EMG device has 16 electrode
    channels, and the signal has been high-pass filtered. Therefore, the fields
    corresponding to left and right EMG are 2D arrays of shape ``(T, 16)`` each
    and ``timestamps`` is a 1D array of length ``T``.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "emg2pose"
    # timeseries keys
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    # metadata keys
    SESSION_NAME: ClassVar[str] = "session"
    SIDE: ClassVar[str] = "side"
    STAGE: ClassVar[str] = "stage"
    START_TIME: ClassVar[str] = "start"
    END_TIME: ClassVar[str] = "end"
    NUM_CHANNELS: ClassVar[str] = "num_channels"
    DATASET_NAME: ClassVar[str] = "dataset"
    USER: ClassVar[str] = "user"
    SAMPLE_RATE: ClassVar[str] = "sample_rate"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2pose_group: h5py.Group = self._file[self.HDF5_GROUP]

        # ``timeseries`` is a HDF5 compound Dataset
        self.timeseries: h5py.Dataset = emg2pose_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG in self.timeseries.dtype.fields
        assert self.JOINT_ANGLES in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # Load the metadata entirely into memory as it's rather small
        self.metadata: dict[str, Any] = {}
        for key, val in emg2pose_group.attrs.items():
            self.metadata[key] = val

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice) -> np.ndarray:
        return self.timeseries[key]

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        """Load and return a contiguous slice of the timeseries windowed
        by the provided start and end timestamps.

        Args:
            start_t (float): The start time of the window to grab
                (in absolute unix time). Defaults to selecting from the
                beginning of the session. (default: ``-np.inf``).
            end_t (float): The end time of the window to grab
                (in absolute unix time). Defaults to selecting until the
                end of the session. (default: ``np.inf``)
        """
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    @property
    def fields(self) -> KeysView[str]:
        """The names of the fields in ``timeseries``."""
        fields: KeysView[str] = self.timeseries.dtype.fields.keys()
        return fields

    @property
    def timestamps(self) -> np.ndarray:
        """EMG timestamps.

        NOTE: This reads the entire sequence of timesetamps from the underlying
        HDF5 file and therefore incurs disk latency. Avoid this in the critical
        path."""
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "Not monotonic"
        return emg_timestamps

    @property
    def session_name(self) -> str:
        """Unique name of the session."""
        return self.metadata[self.SESSION_NAME]

    @property
    def user(self) -> str:
        """Unique ID of the user this session corresponds to."""
        return self.metadata[self.USER]

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.session_name} ({len(self)} samples)"

# %%
# Download the dataset
# --------------------
#

# %%
data_dir = Path.home() / "emg_data"
emg_dir = data_dir / "emg2pose_dataset_mini"
want_fpath = "emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left.hdf5"

unpack = pooch.Untar(extract_dir=data_dir, # Relative to the path where the zip file is downloaded
                     members=[want_fpath]
                     )
emg_fpaths = pooch.retrieve(
    url="https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar",
    known_hash="sha256:d7400e98508ccbb2139c2d78e552867b23501f637456546fd6680f3fe7fec50d",
    progressbar=True,
    path=data_dir,
    processor=unpack,
)
emg_fname = Path(emg_fpaths[0])
emg_dir = emg_fname.parent
# Delete the large tar file
# list(emg_dir.glob("*.tar"))[0].unlink()

# %%
# Load the data
# ----------------
#

# %%
data = Emg2PoseSessionData(hdf5_path=emg_fname)

# %%
# Visualize the data
# ------------------
# We'll let MNE-Python do the heavy lifting for us here.

# %%
ch_names = [f"EMG{ii:02}" for ii, _ in enumerate(data["emg"].T, 1)]
ch_types = ["emg"] * len(ch_names)
sfreq = data.metadata[Emg2PoseSessionData.SAMPLE_RATE]
info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
# MNE expects data in the shape (n_channels, n_times). So we need to transpose the data
raw = mne.io.RawArray(data["emg"].T, info)
# MNE expects the EMG data to be in Volts, so we need to scale it from mV to V
raw.apply_function(lambda x: x * 1e-6, picks="emg")
raw.plot(start=20, duration=20)

# %%
# Use PCA and KMeans to cluster the data
# --------------------------------------
#
# We'll use PCA to reduce the data dimenstionality to 3D
# and then use KMeans to cluster the data.
#

# %%
n_components = 3
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data["emg"])
clusters = KMeans(n_clusters=5).fit_predict(data_pca)

# %%
# Visualize the clusters
# ----------------------
#

# %%
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=clusters)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA of EMG data with KMeans clustering")
plt.show()


