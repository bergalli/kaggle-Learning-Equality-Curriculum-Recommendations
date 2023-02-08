"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import f


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                f,
                inputs=dict(
                    content="content",
                    correlations="correlations",
                    sample_submission="sample_submission",
                    topics="topics",
                ),
                outputs="o",
            )
        ]
    )
