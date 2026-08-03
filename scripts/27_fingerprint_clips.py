"""
Phase 11d runner: are the sixteen clip URLs sixteen different videos?

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

  About 48 small ranged requests plus one full clip download. Two minutes or
  so, depending on how big that clip really is. Needs
  data/interim/video_playback.csv from scripts/26_verify_playback.py.

WHY
  Phase 11c got real MP4 bytes from all sixteen clips, which was the good news,
  and an identical advertised size of 31,580,089 bytes from every one of them,
  which was not. Sixteen videos of 4.4 to 16.6 seconds cannot weigh the same.

  Either that header is meaningless, or every URL is serving the same file. The
  second would be fatal: the panel would show a video that has nothing to do
  with the play it sits beside.

  This hashes three byte ranges in each clip. Different videos have different
  bytes. If all sixteen headers hash the same, the answer is no and the feature
  does not get built.

  The verdict rule is written into src/video_fingerprint.py ahead of the data:
  distinct header hashes must equal the number of clips tested.

READ ONLY
  Writes reports/video_fingerprint.txt and data/interim/video_fingerprint.csv.
  Nothing else is touched.

WHAT TO DO
  Paste the output back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_fingerprint  # noqa: E402

if __name__ == "__main__":
    video_fingerprint.main()
