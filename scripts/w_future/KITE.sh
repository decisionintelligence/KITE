#!/usr/bin/env bash
set -euo pipefail

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list NP.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"horizon":24,"lr":0.002,"noise_dropout":0.2,"rank":64,"rate":8,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path NP/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list NP.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"horizon":360,"lr":0.002,"noise_dropout":0.2,"rank":64,"rate":8,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path NP/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list PJM.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.2,"flow_depth":4,"flow_dim":512,"flow_head":8,"horizon":24,"mlp_ratio":6.0,"noise_dropout":0.25,"omega":0.9,"p_uncond":0.05,"rate":16,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path PJM/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list PJM.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.2,"flow_depth":4,"flow_dim":512,"flow_head":8,"horizon":360,"mlp_ratio":6.0,"noise_dropout":0.25,"omega":0.9,"p_uncond":0.05,"rate":16,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path PJM/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list BE.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"flow_dim":64,"flow_head":2,"horizon":24,"lr":0.0003,"omega":0.8,"p_uncond":0.05,"rate":22,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path BE/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list BE.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"flow_dim":64,"flow_head":2,"horizon":360,"lr":0.0003,"omega":0.8,"p_uncond":0.05,"rate":22,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path BE/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list FR.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"flow_depth":6,"flow_dim":384,"horizon":24,"lr":0.002,"mlp_ratio":3.0,"omega":0.9,"p_uncond":0.15,"rank":64,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path FR/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list FR.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"flow_depth":6,"flow_dim":384,"horizon":360,"lr":0.002,"mlp_ratio":3.0,"omega":0.9,"p_uncond":0.15,"rank":64,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path FR/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list DE.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"flow_depth":5,"flow_head":2,"horizon":24,"lr":0.002,"noise_dropout":0.2,"omega":1.2,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path DE/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list DE.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"flow_depth":5,"flow_head":2,"horizon":360,"lr":0.002,"noise_dropout":0.2,"omega":1.2,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path DE/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Energy.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"flow_depth":3,"flow_dim":512,"horizon":24,"lr":0.001,"noise_dropout":0.25,"omega":0.9,"rank":32,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Energy/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Energy.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.1,"flow_depth":3,"flow_dim":512,"horizon":360,"lr":0.001,"noise_dropout":0.25,"omega":0.9,"rank":32,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Energy/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfm1.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"flow_depth":4,"flow_head":8,"horizon":24,"noise_dropout":0.2,"rate":16,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfm1/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfm1.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"flow_depth":4,"flow_head":8,"horizon":360,"noise_dropout":0.2,"rate":16,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfm1/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfm2.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.2,"flow_head":8,"horizon":24,"noise_dropout":0.2,"rate":22,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfm2/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfm2.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.2,"flow_head":8,"horizon":360,"noise_dropout":0.2,"rate":22,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfm2/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfh1.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"flow_dim":64,"horizon":24,"lr":0.0005,"noise_dropout":0.275,"p_uncond":0.15,"rate":2,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfh1/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfh1.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"flow_dim":64,"horizon":360,"lr":0.0005,"noise_dropout":0.275,"p_uncond":0.15,"rate":2,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfh1/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfh2.csv --strategy-args '{"horizon":24,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.2,"flow_depth":4,"flow_dim":64,"horizon":24,"lr":0.00036,"noise_dropout":0.35,"omega":1.1,"p_uncond":0.2,"rank":16,"rate":10,"seq_len":168,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfh2/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Sdwpfh2.csv --strategy-args '{"horizon":360,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"dropout":0.2,"flow_depth":4,"flow_dim":64,"horizon":360,"lr":0.00036,"noise_dropout":0.35,"omega":1.1,"p_uncond":0.2,"rank":16,"rate":10,"seq_len":720,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Sdwpfh2/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Colbun.csv --strategy-args '{"horizon":10,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"batch_size":64,"flow_depth":6,"horizon":10,"noise_dropout":0.0,"omega":1.2,"rank":48,"rate":4,"seq_len":60,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Colbun/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Colbun.csv --strategy-args '{"horizon":30,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"batch_size":64,"flow_depth":6,"horizon":30,"noise_dropout":0.0,"omega":1.2,"rank":48,"rate":4,"seq_len":180,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Colbun/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Rapel.csv --strategy-args '{"horizon":10,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"batch_size":64,"dropout":0.3,"flow_depth":3,"flow_head":2,"horizon":10,"lr":0.001,"omega":0.9,"p_uncond":0.15,"rank":64,"seq_len":60,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Rapel/KITE

python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list Rapel.csv --strategy-args '{"horizon":30,"target_channel":[-1]}' --model-name kite.KITE --model-hyper-params '{"batch_size":64,"dropout":0.3,"flow_depth":3,"flow_head":2,"horizon":30,"lr":0.001,"omega":0.9,"p_uncond":0.15,"rank":64,"seq_len":180,"use_future_exog":true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path Rapel/KITE
