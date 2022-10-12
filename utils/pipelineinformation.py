from dataclasses import dataclass

from sklearn.pipeline import Pipeline

@dataclass()
class PipelineInformation:
    pipeline: Pipeline
    classification: bool # True if classification task, else false