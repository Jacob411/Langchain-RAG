import pytest
from query import query  # Replace with your actual function call

@pytest.mark.parametrize(
    "question, keywords",
    [
        ("Who do I need to contact for the ai chatbot project", ["darren may", "christine cline"]),
        ("what is jaunt?", ["golf", "cart"]),
        ("What expertise is required for the ai chatbot?", ["ai", "machine learning"]),
        ("What is tiger check?", ['health', 'children'])
    ]
)
@pytest.mark.filterwarnings("ignore:PyType_Spec")
def test_keywords_in_response(question, keywords):
  """
  Tests if keywords are present in the response for a given question.
  """

  response = query(question).lower()  # Get response and convert to lowercase
  for keyword in keywords:
    assert keyword in response, f"Keyword '{keyword}' not found in response to '{question}'"