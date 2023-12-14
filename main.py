from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

# return_messages=True formats the messages
# for chat-based models instead of regular strings
# i.e. HumanMessage, AIMessage
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("memory/messages.json"),
    memory_key="messages",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        # Tells LangChain to look in the memory input
        # for a variable named "messages"
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input("> ")
    result = chain({"content": content})
    print(result["text"])
