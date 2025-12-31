## Make dataset (Raw -> Interim)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) telco_customer_churn_prediction/dataset.py

## Generate Features (Interim -> Processed)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) telco_customer_churn_prediction/features.py

## Train Model (Processed -> Model)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) telco_customer_churn_prediction/modeling/train.py

## Run Full Pipeline (Data -> Features -> Train)
.PHONY: pipeline
pipeline: data features train