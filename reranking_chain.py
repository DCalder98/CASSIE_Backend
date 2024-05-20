from langchain import OpenAI
from reranking_template import RerankingTemplate

class Reranker:
    @staticmethod
    def generate(query: str, passages: str, k: int) -> list[str]:
        llm = OpenAI(model_name="gpt-4o", temperature=0)  # Use your OpenAI API key here
        template = RerankingTemplate().create_template()
        chain = ...  # Build your chain here with the template and LLM

        response = chain.invoke({
            "question": query,
            "passages": passages,
            "k": k
        })

        return response["rerank"]