from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI

class RerankingTemplate(BasePromptTemplate):
    def __init__(self):
        self.prompt = """You are an AI language model assistant. Your task is to rerank passages related to a query based on their relevance. The most relevant passages should be put at the beginning. You should only pick at max {k} passages.
        The following are passages related to this query: {question}.
        Passages: {passages}"""

    def format(self, **kwargs) -> str:
        return self.prompt.format(**kwargs)

    def format_prompt(self, **kwargs) -> str:
        return self.prompt.format(**kwargs)

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages", "k"]
        )

class Reranker:
    @staticmethod
    def generate(query: str, passages: str, k: int) -> list[str]:
        llm = OpenAI(model_name="gpt-4o", temperature=0)  # Use your OpenAI API key here
        template = RerankingTemplate().create_template()
        chain = LLMChain(llm=llm, prompt=template, output_key="rerank")
        
        response = chain.invoke({
            "question": query,
            "passages": passages,
            "k": k
        })

        return response["rerank"]
