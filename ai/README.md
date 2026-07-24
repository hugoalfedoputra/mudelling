# What to change to run

## Training/retraining

1. **HSA override GFX version** on `train.py` and `system_check.py`
2. **params.json** (copy paste params.example.json and change the name)
3. **.env** (when TESTING CODE, change the validation and/or test CSVs)
4. **train.py** (to manually override function call at `main()` and change global variables)

## Tuning

1. **HSA override GFX version** on `train.py` and `system_check.py`
2. **.env** (when TESTING CODE, change the validation and/or test CSVs)
3. **train.py** (to manually override function call at `main()` and change global variables)

## Testing

1. **HSA override GFX version** on `train.py` and `system_check.py`
2. **.env** (when TESTING CODE, change the validation and/or test CSVs)
3. **testing.py** (globals and hardcoded filter values)

## No uv

```
pip install --ignore-installed blinker cryptography
```

then

```
pip install "matplotlib>=3.10.9" "mlflow>=1.27.0" "pandas>=3.0.3" "protobuf<5" "python-dotenv>=1.2.2" "rclone-python>=0.1.24" "scikit-learn>=1.8.0" scipy seaborn torchinfo
```

or

```
pip install "matplotlib>=3.10.9" "mlflow>=1.27.0" pandas "protobuf<5" "python-dotenv>=1.2.2" "rclone-python>=0.1.24" "scikit-learn>=1.8.0" scipy seaborn torchinfo
```

## mlflow_pgdump

```
docker exec -it mlflow-postgres psql -U ${USER} -d ${DB} -c "COPY (SELECT * FROM metrics WHERE run_uuid='blabla' ORDER BY step) TO STDOUT WITH CSV HEADER" > blabla_history.csv
```

```
scp -i /path/to/pk server@123.456.123.456:/path/to/blabla_history.csv /path/to/skripsi-code/ai/mlflow_pgdump/blabla_history.csv
```
