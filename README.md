# Localised Models for Aqueous Solubility Prediction

## Available Functions
Run local model hyperparameter sweep
`python main.py local_hyperopt`

Run global model hyperparameter sweep
`python main.py global_hyperopt`

Train and save a global model
`python main.py train_global_model`

Run global model 5-fold cross validation
`python main.py global_cv`

Test local model configuration
`python main.py local_test`

Make a prediction from the global model
`python main.py global_predict <smiles>`

Make a prediction from the local model
`python main.py global_predict <smiles>`
