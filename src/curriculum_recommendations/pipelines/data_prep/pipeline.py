"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_dataframe_length, read_data, get_embeddings, train_topic_assignment_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                read_data,
                inputs=dict(
                    content="content",
                    # sample_submission="sample_submission",
                    topics="topics",
                ),
                outputs=["topics_cleaned", "content_cleaned"],
            ),
            node(
                filter_dataframe_length,
                inputs=dict(
                    content="content_cleaned",
                    correlations="correlations",
                    sample_submission="sample_submission",
                    topics="topics_cleaned",
                    perc_of_rows_to_keep="params:perc_of_rows_to_keep"
                ),
                outputs=["topics_filtered", "content_filtered"],
            ),
            node(
                get_embeddings,
                inputs=dict(topics_cleaned="topics_filtered", content_cleaned="content_filtered"),
                outputs=["topics_embeddings", "content_embeddings"],
            ),
            node(
                train_topic_assignment_model,
                inputs=dict(
                    topics_embeddings="topics_embeddings",
                    content_embeddings="content_embeddings",
                    topics_cleaned="topics_cleaned",
                    content_cleaned="content_cleaned",
                    sample_submission="sample_submission",
                ),
                outputs="submission_topics_assigned"
            ),
        ]
    )
