from pathlib import Path
from functools import partial

import polars as pl
from datasets import load_dataset

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import flytekit
from flytekit import task, dynamic, ImageSpec
import flytekit.deck
from flytekitplugins.spark import DatabricksV2 as Databricks
from flytekit.types.file import FlyteFile


image = ImageSpec(
    builder="envd",
    registry="ghcr.io/granthamtaylor",
    name='byoc-sandbox',
    base_image="ghcr.io/unionai-oss/databricks:kmeans",
    source_root=".",
    packages = [
        "pyspark",
        "polars",
        "pyarrow",
        "datasets",
        "numpy",
        "union",
        "flytekitplugins-spark>=1.13.1a2",
        "plotly",
        "flytekit>=1.13.1a2",
    ],
)

databricks = Databricks(
    spark_conf={
        "spark.driver.memory": "1000M",
        "spark.executor.memory": "1000M",
        "spark.executor.cores": "1",
        "spark.executor.instances": "2",
        "spark.driver.cores": "1",
        "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar",
    },
    databricks_conf={
        "run_name": "kmeans clustering example",
        "runtime_engine": "PHOTON",
        "new_cluster": {
            "spark_version": "14.3.x-scala2.12",
            "node_type_id": "r6id.xlarge",
            "num_workers": 3,
            "aws_attributes": {
                "availability": "SPOT_WITH_FALLBACK",
                "instance_profile_arn": "arn:aws:iam::339713193121:instance-profile/databricks-demo",
                "first_on_demand": 1,
                "zone_id": "auto",
            },
        },
        "timeout_seconds": 3600,
        "max_retries": 3,
    },
    databricks_instance="dbc-ca63b07f-c54a.cloud.databricks.com",
)

@task(task_config=databricks, container_image=image)
def train(data: FlyteFile, params: dict[str, float|int]) -> float:
    
    session = flytekit.current_context().spark_session
    dataframe = session.read.parquet(data.remote_source) 
    
    label = 'agep'

    feature_columns = [col for col in dataframe.columns if col != label]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data_assembled = assembler.transform(dataframe)

    train_data, test_data = data_assembled.randomSplit([0.7, 0.3], seed=42)

    gbt = GBTRegressor(
        labelCol=label, 
        maxDepth=params['maxDepth'],
        maxBins=params['maxBins'],
        maxIter=params['maxIter'],
        stepSize=params['stepSize'],
    )

    model = gbt.fit(train_data)

    predictions = model.transform(test_data)

    evaluator = RegressionEvaluator(
        labelCol=label,
        predictionCol="prediction", 
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    print(f"RMSE on test data = {rmse}")

    feature_importance = model.featureImportances
    for i, imp in enumerate(feature_importance):
        print(f"Feature {feature_columns[i]} importance: {imp}")
    
    return rmse


@task(container_image=image)
def generate_data(dataset: str) -> FlyteFile:
    
    datapath = Path(flytekit.current_context().working_directory) / "dataset.parquet"

    ds = load_dataset("birkhoffg/folktables-acs-income")
    
    # 'AGEP': Age
    # 'COW': Class of worker
    # 'SCHL': Educational attainment
    # 'MAR': Marital status
    # 'OCCP': Occupation Code
    # 'POBP': Place of Birth
    # 'RELP': Religious affiliation
    # 'WKHP': Average Weekly Hours Worked
    # 'SEX': Sex
    # 'RAC1P': Racial background
    # 'STATE': State
    # 'YEAR': Year
    # 'PINCP: Personal Income

    
    df = pl.from_arrow(ds.data['train'].table)

    
    df.columns = [col.lower() for col in df.columns]
    
    (
        df
        .select(pl.exclude("__index_level_0__", "state"))
        .write_parquet(datapath)
    )

    return FlyteFile(str(datapath))


@dynamic(container_image=image)
def demo(dataset: str = 'hf://datasets/dipamc/loan_customer/Training Data 2.csv'):
    
    grid = [
        {"maxDepth": 5, "maxBins": 32, "maxIter": 20, "stepSize": 0.2},
        {"maxDepth": 6, "maxBins": 48, "maxIter": 30, "stepSize": 0.2},
        {"maxDepth": 7, "maxBins": 64, "maxIter": 40, "stepSize": 0.2},
        {"maxDepth": 7, "maxBins": 64, "maxIter": 80, "stepSize": 0.1},
    ]
    
    data = generate_data(dataset=dataset)
    
    for params in grid:
        train(data=data, params=params)
