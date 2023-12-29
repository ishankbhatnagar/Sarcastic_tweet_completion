from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

HUGGINGFACEHUB_API_TOKEN = ""

hub_llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={"temperature": 0.8, "max_length": 100},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

template = (
    "You had one job! You are the {profession} and you didn't have to be sarcastic"
)

input_variables = ["profession"]

prompt = PromptTemplate(template=template, input_variables=input_variables)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

user_profession = input("Enter your profession: ")

print(hub_chain.run(user_profession))
