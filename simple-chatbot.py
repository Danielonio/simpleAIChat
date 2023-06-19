from dotenv import load_dotenv
load_dotenv() 
product_metadata = [
{
'item_id': 'B07T6RZ2CM',
 'marketplace': 'Amazon',
 'country': 'IN',
 'main_image_id': '71dZhpsferL',
 'domain_name': 'amazon.in',
 'bullet_point': '3D Printed Hard Back Case Mobile Cover for Lenovo K4 Note Easy to put & take off with perfect cutouts for volume buttons, audio & charging ports. Stylish design and appearance, express your unique personality. Extreme precision design allows easy access to all buttons and ports while featuring raised bezel to life screen and camera off flat surface. Slim Hard Back Cover No Warranty None',
 'item_keywords': 'mobile cover back cover mobile case phone case mobile panel phone panel Lenovo mobile case Lenovo phone cover Lenovo back case hard case 3D printed mobile cover mobile cover back cover mobile case phone case mobile panel phone panel Lenovo mobile case Lenovo phone cover Lenovo back case hard case 3D printed mobile cover mobile cover back cover mobile case phone case mobile panel phone panel Lenovo mobile case Lenovo phone cover Lenovo back case hard case 3D printed mobile cover mobile cover back cover mobil',
 'material': 'plastic',
 'brand': 'Amazon Brand - Solimo',
 'color': 'Others',
 'item_name': 'Amazon Brand - Solimo Designer Couples Sitting at Dark 3D Printed Hard Back Case Mobile Cover for Lenovo K4 Note',
 'model_name': 'Lenovo K4 Note',
 'model_number': 'gz8115-SL40423',
 'product_type': 'CELLULAR_PHONE_CASE',
 'primary_key': 'B07T6RZ2CM-amazon.in'
 },
 {'item_id': 'B08D494G9F',
 'marketplace': 'Amazon',
 'country': 'IN',
 'main_image_id': '7121v7PBWkL',
 'domain_name': 'amazon.in',
 'bullet_point': 'Outer Material: Synthetic Closure Type: Slip On Heel type: flats Toe Style: Open Toe Warranty Type: Manufacturer Warranty Description: 45 days',
 'item_keywords': 'Elise printed thong flip flops upper cushioned footbed patterned rubber',
 'material': 'leather',
 'brand': 'ELISE',
 'color': 'Multi',
 'item_name': "ELISE Women's Multi Flip-Flops-5 UK (38 EU) (6 US) (EFFS20-15)",
 'model_name': 'the shoes 45',
 'model_number': 'EFFS20-15',
 'product_type': 'SHOES',
 'primary_key': 'B08D494G9F-amazon.in'
 }
 ]

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore

# data that will be embedded and converted to vectors
texts = [
    v['item_name'] for v in product_metadata
]

print(texts)
# product metadata that we'll store along our vectors
metadatas = product_metadata

print(metadatas)
# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()

# name of the Redis search index to create
index_name = "products"

# assumes you have a redis stack server running on within your docker compose network
redis_url = "redis://localhost:6379/"

# create and load redis with documents
vectorstore = RedisVectorStore.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embedding,
    index_name=index_name,
    redis_url=redis_url
)

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.

Chat History:\"""
{chat_history}
\"""

Follow Up Input: \"""
{question}
\"""

Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(template)

template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.
It's ok if you don't know the answer.

Context:\"""
{context}
\"""

Question:\"
\"""

Helpful Answer:"""

qa_prompt= PromptTemplate.from_template(template)


# define two LLM models from OpenAI
llm = OpenAI(temperature=0)

streaming_llm = OpenAI(
    streaming=True,
    callback_manager=CallbackManager([
        StreamingStdOutCallbackHandler()]),
    verbose=True,
    temperature=0.2,
    max_tokens=150
)

# use the LLM Chain to create a question creation chain
question_generator = LLMChain(
    llm=llm,
    prompt=condense_question_prompt
)

# use the streaming LLM to create a question answering chain
doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
)


chatbot = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)

# create a chat history buffer
chat_history = []

# gather user input for the first question to kick off the bot
question = input("Hi! What are you looking for today?")

# keep the bot running in a loop to simulate a conversation
while True:
    result = chatbot(
        {"question": question, "chat_history": chat_history}
    )
    print("\n")
    chat_history.append((result["question"], result["answer"]))
    question = input()