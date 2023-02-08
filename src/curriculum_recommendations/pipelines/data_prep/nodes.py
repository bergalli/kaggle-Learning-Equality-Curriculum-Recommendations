"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""


def f(
    content,
    correlations,
    sample_submission,
    topics,
):
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return False
