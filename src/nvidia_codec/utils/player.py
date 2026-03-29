"""Convenience player: opens a file, picks the best video track, decodes."""

from .demux import parse, VideoTrack
from .decode import Decoder


class Player(Decoder):
    """Convenience class: opens a file, picks the best video track, decodes.

    Example:
        with Player('/path/to/video.mp4') as player:
            t, frame = player.screenshot(timedelta(seconds=30), torch.uint8)
    """

    def __init__(
        self,
        url,
        target_size=None,
        cropping=None,
        target_rect=None,
        device=None,
        track_idx=None,
    ):
        tracks = [t for t in parse(url) if isinstance(t, VideoTrack)]
        if not tracks:
            raise ValueError(f"{url} has no video stream")
        if track_idx is not None:
            track = next((t for t in tracks if t.index == track_idx), None)
            if not track:
                raise ValueError(f"{url} has no video track with index {track_idx}")
        elif len(tracks) == 1:
            track = tracks[0]
        else:
            raise ValueError(f"{url} has {len(tracks)} video tracks, specify track_idx")
        super().__init__(
            track,
            target_size=target_size,
            cropping=cropping,
            target_rect=target_rect,
            device=device,
        )
        self.url = url
