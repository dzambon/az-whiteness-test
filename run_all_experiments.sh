################################################
# Train the models                             #
################################################

python tsl_experiments.py --dataset-name la --model-name fcrnn --config fcrnn.yaml
python tsl_experiments.py --dataset-name la --model-name dcrnn --config dcrnn.yaml
python tsl_experiments.py --dataset-name la --model-name gwnet --config gwnet.yaml
python tsl_experiments.py --dataset-name la --model-name gatedgn --config gatedgn.yaml

python tsl_experiments.py --dataset-name bay --model-name fcrnn --config fcrnn.yaml
python tsl_experiments.py --dataset-name bay --model-name dcrnn --config dcrnn.yaml
python tsl_experiments.py --dataset-name bay --model-name gwnet --config gwnet.yaml
python tsl_experiments.py --dataset-name bay --model-name gatedgn --config gatedgn.yaml

python tsl_experiments.py --dataset-name gpolyvar --model-name fcrnn --config fcrnn.yaml
python tsl_experiments.py --dataset-name gpolyvar --model-name dcrnn --config dcrnn.yaml
python tsl_experiments.py --dataset-name gpolyvar --model-name gwnet --config gwnet.yaml
python tsl_experiments.py --dataset-name gpolyvar --model-name gatedgn --config gatedgn.yaml