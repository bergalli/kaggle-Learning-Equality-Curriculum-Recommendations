"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_topics, read_data, get_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     clean_topics,
            #     inputs=dict(
            #         content="content",
            #         correlations="correlations",
            #         sample_submission="sample_submission",
            #         topics="topics",
            #     ),
            #     outputs="o",
            # ),
            node(
                read_data,
                inputs=dict(
                    content="content",
                    sample_submission="sample_submission",
                    topics="topics",
                ),
                outputs=["clean_topics", "clean_content"],
            ),
            node(
                get_embeddings,
                inputs=dict(tmp_topics="clean_topics", tmp_content="clean_content"),
                outputs=["topics_preds", "content_preds"],
            ),
        ]
    )
