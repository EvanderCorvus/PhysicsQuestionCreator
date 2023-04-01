import openai
import os
import langchain as lc
import llm_utilities as utils
from langchain.llms import OpenAI

key = os.getenv('OPENAI_API_KEY')
openai.api_key = key

llm = utils.get_llm()

fname = 'prompt'
template = utils.get_template(fname)

fname = 'laws'
laws = utils.get_template(fname)

fname = 'examples'
examples = utils.get_template(fname)

prompt = lc.PromptTemplate(input_variables=['laws','examples'],template=template)

llm_chain = lc.LLMChain(prompt=prompt,llm=llm)

print(llm_chain.run(laws=laws,examples=examples))


