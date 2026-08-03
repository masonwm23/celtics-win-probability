"""
Phase 11e runner: why does every clip URL return the same file?

HOW TO RUN IN SPYDER
  Open this file and press F5 (Run file).

  About 30 small ranged requests. Well under a minute. Needs
  data/interim/video_probe.csv from scripts/24_probe_video.py.

WHERE WE ARE
  Phase 11d hashed sixteen genuinely distinct clip URLs across eight seasons.
  All sixteen returned byte-identical content, 31,580,089 bytes. videos.nba.com
  is serving one file for every address.

  That does NOT yet mean the idea is dead. Three things decide it, and two of
  them would change the answer:

    1. Does a FABRICATED url, real in shape but naming a uuid that does not
       exist, return the same file? If yes, the host answers everything with a
       default and the 200s never meant anything.

    2. Are the still images gated the same way? Every matched play carries a
       thumbnail, and all 336 are distinct URLs. If the stills come back
       distinct and well-formed, a panel showing the real frame of the real
       play is still buildable, which is most of what you asked for.

    3. Is only the 960x540 encoding affected? The same play is also published
       at 320x180.

  Each URL is tried with three header sets, because the difference between
  "blocked" and "broken" shows up there.

  The verdict rules are written into src/video_mechanism.py ahead of the data.

READ ONLY
  Writes reports/video_mechanism.txt and data/interim/video_mechanism.csv.
  Nothing else is touched.

WHAT TO DO
  Paste the output back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import video_mechanism  # noqa: E402

if __name__ == "__main__":
    video_mechanism.main()
