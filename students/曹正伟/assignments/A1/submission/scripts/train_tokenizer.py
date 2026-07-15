from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.train_tokenizer import main

if __name__ == "__main__":
    main()
