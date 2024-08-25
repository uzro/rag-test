import argparse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from create_database import CHROMA_PATH

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str)
  args = parser.parse_args()

  query_text = args.input

  # prepare the DB
  embedding_function = OpenAIEmbeddings()
  db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH)

  # Search the DB
  results = db.similarity_search_with_relevance_scores(query_text)
  if len(results) == 0 or results[0][1] < 0.7:
    print("No results found")
    return

  context_text = "\n\n-----\n\n".join([doc.page_content for doc, _ in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  print(prompt)

  model = ChatOpenAI(model_name="gpt-3.5-turbo")
  response_text = model.predict(prompt)

  sources = [doc.metadata.get("source", None) for doc, _ in results]
  formatted_response = f"Response: {response_text}\n\nSources: {sources}"
  print(formatted_response)


if __name__ == "__main__":
  main()
