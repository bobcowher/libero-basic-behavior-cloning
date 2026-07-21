source ~/anaconda3/etc/profile.d/conda.sh

conda activate libero-modern

# Run once
# python $(python -c "import robosuite, os; print(os.path.dirname(robosuite.__file__))")/scripts/setup_macros.py
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"
export ROBOSUITE_LOG_LEVEL=ERROR   # robosuite uses its own logger, not warnings

#pip install -r requirements.txt

python -u ./train.py
