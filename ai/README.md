# What to change to run

## Training/retraining
1. **params.json** (copy paste params.example.json and change the name)
2. **.env** (when TESTING CODE, change the validation and/or test CSVs)
3. **pyproject.toml** (hacky/manual-override way to install torch cross-GPUs)
4. **train.py** (to manually override function call at `main()` and change global variables)

## Tuning
1. **.env** (when TESTING CODE, change the validation and/or test CSVs)
2. **pyproject.toml** (hacky/manual-override way to install torch cross-GPUs)
3. **train.py** (to manually override function call at `main()` and change global variables)

## Testing
1. **.env** (when TESTING CODE, change the validation and/or test CSVs)
2. **testing.py** (globals and hardcoded filter values)