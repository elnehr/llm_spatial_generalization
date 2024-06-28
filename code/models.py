from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
    retry_if_exception
)  # for exponential backoff
from langchain.llms import HuggingFaceHub
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

oa_client = AsyncOpenAI()
client = AsyncAnthropic()

def log_retry(retry_state):
    # Log the exception message
    logger.error(f"Retry {retry_state.attempt_number} for {retry_state.fn.__name__}, due to exception: {retry_state.outcome.exception()}")

def my_retry_predicate(exception):
    if isinstance(exception, Exception):
        logger.warning(f"Caught exception: {exception}")
        return True
    return False



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
async def async_claude_instant(prompt, user_input):
    response = await client.messages.create(
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": prompt + """ 
                
                """ + user_input,
            }
        ],
        model="claude-instant-1.2",
    )
    return response.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
async def async_claude_3_sonnet(prompt, user_input):
    response = await client.messages.create(
        max_tokens=2048,
        system= prompt,
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
        model="claude-3-sonnet-20240229",
    )
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
async def async_claude_3_opus(prompt, user_input):
    response = await client.messages.create(
        max_tokens=2048,
        system= prompt,
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
        model="claude-3-opus-20240229",
    )
    return response.choices[0].message.content



#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
@retry(wait=wait_random_exponential(min=1, max=60),
       stop=stop_after_attempt(50))
async def async_gpt_3_5_turbo(prompt, user_input):
    response = await oa_client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    )
    return response.choices[0].message.content



#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
async def async_gpt_4_turbo(prompt, user_input):
    response = await oa_client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    )
    return response.choices[0].message.content
                        
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def gemma_7b_it(prompt, question):
    llm = HuggingFaceHub(
        repo_id="google/gemma-7b-it", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""
    <bos><start_of_turn>user
    {prompt}
    
    {question}<end_of_turn>
    <start_of_turn>model
    """

    response = llm.predict(prompt).split("<start_of_turn>model")[-1]
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def gemma_2b_it(prompt, question):
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b-it", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""
    <bos><start_of_turn>user
    {prompt}
    
    {question}<end_of_turn>
    <start_of_turn>model
    """

    response = llm.predict(prompt).split("<start_of_turn>model")[-1]
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def gpt_4_turbo(prompt, question):
    chat_answers = ChatOpenAI(temperature=0, openai_api_key="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", model_name="gpt-4-0125-preview")

    answer_system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
    answer_human_template = f"""{question}"""
    answer_human_message_prompt = HumanMessagePromptTemplate.from_template(answer_human_template)

    answer_chat_prompt = ChatPromptTemplate.from_messages(
        [answer_system_message_prompt, answer_human_message_prompt]
    )

    return chat_answers(answer_chat_prompt.format_prompt(question=question).to_messages()).content


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def gpt_3_5_turbo(prompt, question):
    chat_answers = ChatOpenAI(temperature=0, openai_api_key="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")

    answer_system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
    answer_human_template = f"""{question}"""
    answer_human_message_prompt = HumanMessagePromptTemplate.from_template(answer_human_template)

    answer_chat_prompt = ChatPromptTemplate.from_messages(
        [answer_system_message_prompt, answer_human_message_prompt]
    )

    return chat_answers(answer_chat_prompt.format_prompt(question=question).to_messages()).content


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def mistral_7b(prompt, question):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""
    <s>[INST] 
    {prompt}
    
    {question} [/INST]
    """

    response = llm.predict(prompt).split("[/INST]")[-1]
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def mistral_8x7b(prompt, question):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""
    <s>[INST] 
    {prompt}
    
    {question} [/INST]
    """

    response = llm.predict(prompt).split("[/INST]")[-1]
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def llama2_7b(prompt, question):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""<s>[INST] <<SYS>>
{prompt}
<</SYS>>
{question} [/INST]"""

    response = llm.predict(prompt).split("[/INST]")[-1]
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def llama2_7b(prompt, question):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""<s>[INST] <<SYS>>
{prompt}
<</SYS>>
{question} [/INST]"""

    response = llm.predict(prompt).split("[/INST]")[-1]
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def llama2_13b(prompt, question):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-13b-chat-hf", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""<s>[INST] <<SYS>>
{prompt}
<</SYS>>
{question} [/INST]"""

    response = llm.predict(prompt).split("[/INST]")[-1]
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def llama2_70b(prompt, question):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-70b-chat-hf", 
        model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":512}
    )
    prompt = f"""<s>[INST] <<SYS>>
{prompt}
<</SYS>>
{question} [/INST]"""

    response = llm.predict(prompt).split("[/INST]")[-1]
    return response

    


