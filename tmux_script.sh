source .venv/bin/activate

# Start a new tmux session
tmux new-session -d -s mysession

# Create windows and run scripts
tmux new-window -t mysession:0 -n 0_20 'python 3_generate_hatecheck.py --dataset_name "0_20" > output_0_20.log 2>&1'
tmux new-window -t mysession:1 -n 20_40 'python 3_generate_hatecheck.py --dataset_name "20_40" > output_20_40.log 2>&1'
tmux new-window -t mysession:2 -n 40_60 'python 3_generate_hatecheck.py --dataset_name "40_60" > output_40_60.log 2>&1'
tmux new-window -t mysession:3 -n 60_80 'python 3_generate_hatecheck.py --dataset_name "60_80" > output_60_80.log 2>&1'
tmux new-window -t mysession:4 -n 80_100 'python 3_generate_hatecheck.py --dataset_name "80_100" > output_80_100.log 2>&1'
tmux new-window -t mysession:5 -n 100_120 'python 3_generate_hatecheck.py --dataset_name "100_120" > output_100_120.log 2>&1'
tmux new-window -t mysession:6 -n 120_140 'python 3_generate_hatecheck.py --dataset_name "120_140" > output_120_140.log 2>&1'
tmux new-window -t mysession:7 -n 140_160 'python 3_generate_hatecheck.py --dataset_name "140_160" > output_140_160.log 2>&1'
tmux new-window -t mysession:8 -n 160_180 'python 3_generate_hatecheck.py --dataset_name "160_180" > output_160_180.log 2>&1'
tmux new-window -t mysession:9 -n 180_200 'python 3_generate_hatecheck.py --dataset_name "180_200" > output_180_200.log 2>&1'
tmux new-window -t mysession:10 -n 200_220 'python 3_generate_hatecheck.py --dataset_name "200_220" > output_200_220.log 2>&1'
tmux new-window -t mysession:11 -n 220_240 'python 3_generate_hatecheck.py --dataset_name "220_240" > output_220_240.log 2>&1'

# Attach to the tmux session to monitor the scripts
tmux attach-session -t mysession
