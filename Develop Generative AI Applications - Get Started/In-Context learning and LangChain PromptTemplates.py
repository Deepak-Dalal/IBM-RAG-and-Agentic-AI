# %%capture
# !pip install "ibm-watsonx-ai==1.0.8" --user
# !pip install "langchain==0.2.11" --user
# !pip install "langchain-ibm==0.1.7" --user
# !pip install "langchain-core==0.2.43" --user

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# IBM WatsonX imports
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain  # Still using this for backward compatibility

def llm_model(prompt_txt, params=None):
    
    model_id = "ibm/granite-3-2-8b-instruct"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    # Set up credentials for WatsonxLLM
    url = "https://us-south.ml.cloud.ibm.com"
    api_key = "your api key here"
    project_id = "skills-network"

    credentials = {
        "url": url,
        # "api_key": api_key
        # uncomment the field above and replace the api_key with your actual Watsonx API key
    }
    
    # Create LLM directly
    granite_llm = WatsonxLLM(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=default_params
    )
    
    response = granite_llm.invoke(prompt_txt)
    return response

# GenParams().get_example_values()

# Exercise 1
# Experiment with different basic prompts by changing the input phrase. Try these prompts and compare the different responses:

# "The future of artificial intelligence is"
# "Once upon a time in a distant galaxy"
# "The benefits of sustainable energy include"

params = {
    "max_new_tokens": 128, # Try 256 or 512 for more detailed answers
    "min_new_tokens": 10, # Increase to 25-50 if you want more substantial answers
    "temperature": 0.5, # Controls randomness in generation (0.0-1.0)
                       # Lower (0.1-0.3): More focused, consistent, factual responses
                      # Higher (0.7-1.0): More creative, diverse, unpredictable outputs
    "top_p": 0.2, # Nucleus sampling - considers only highest probability tokens
                       # Lower values (0.1-0.3): More conservative, focused text
                       # Higher values (0.7-0.9): More diverse vocabulary and ideas
    "top_k": 1 # Limits token selection to top k most likely tokens
                       # 1 = greedy decoding (always picks most likely token)
                       # Try 40-50 for more varied outputs
}

# Compare responses to different prompts
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The benefits of sustainable energy include"
]

for prompt in prompts:
    response = llm_model(prompt, params)
    print(f"prompt: {prompt}\n")
    print(f"response : {response}\n")

prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'

            Answer:
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")

# Exercise 2
# Create zero-shot prompts for the following tasks:

# Write a prompt that asks the model to classify a movie review as positive or negative.
# Create a prompt that instructs the model to summarize a paragraph about climate change.
# Design a prompt that asks the model to translate an English phrase to Spanish without showing any examples.

# 1. Prompt for Movie Review Classification
movie_review_prompt = """
Classify the following movie review as either 'positive' or 'negative'.

Review: "I was extremely disappointed by this film. The plot was predictable, the acting was wooden, and the special effects looked cheap. I can't recommend this to anyone."

Classification:
"""

# 2. Prompt for Climate Change Paragraph Summarization
climate_change_prompt = """
Summarize the following paragraph about climate change in no more than two sentences.

Paragraph: "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases. The consequences of climate change include more frequent and severe droughts, storms, and heat waves, rising sea levels, melting glaciers, and warming oceans which can directly impact biodiversity, agriculture, and human health."

Summary:
"""

# 3. Prompt for English to Spanish Translation
translation_prompt = """
Translate the following English phrase into Spanish.

English: "I would like to order a coffee with milk and two sugars, please."

Spanish:
"""

responses = {}
responses["movie_review"] = llm_model(movie_review_prompt)
responses["climate_change"] = llm_model(climate_change_prompt)
responses["translation"] = llm_model(translation_prompt)

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()