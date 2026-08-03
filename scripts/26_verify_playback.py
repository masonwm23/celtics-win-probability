"""
Phase 11c runner: check the clips are really there, not just answering HEAD.

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

  About 32 requests, downloads roughly 32 KB in total. Well under a minute.
  Needs data/interim/video_probe.csv, which scripts/24_probe_video.py wrote.

WHY
  Phase 11 reported 40 out of 40 clips at HTTP 200 via HEAD, median size
  31.6 MB. The row-level CSV shows all forty reported EXACTLY 31.6 MB, for
  clips lasting anywhere from 4.4 to 16.6 seconds. Forty different videos
  cannot be the same number of bytes. The HEAD was being answered by something
  in front of the file, so it measured nothing.

  This asks for the first 2 KB of each clip and checks the bytes contain the
  MPEG-4 `ftyp` box. A CDN that is merely being agreeable about headers cannot
  produce those.

READ ONLY
  Writes reports/video_playback.txt and data/interim/video_playback.csv.
  Nothing else is touched.

WHAT TO DO
  Paste the output back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_playback_check  # noqa: E402

if __name__ == "__main__":
    video_playback_check.main()
