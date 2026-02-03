source .venv/bin/activate
pkill -f run_router
ray stop --force
rm -rf core*