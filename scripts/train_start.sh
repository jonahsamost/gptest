uv sync
source .venv/bin/activate

python -m gptest.data.dataset -n 80 -w 10
python -m gptest.scripts.train