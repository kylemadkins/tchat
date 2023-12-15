from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

# return_messages=True formats the messages
# for chat-based models instead of regular strings
# i.e. HumanMessage, AIMessage
memory = ConversationSummaryMemory.from_messages(
    llm=chat,
    chat_memory=FileChatMessageHistory("memory/messages.json"),
    return_messages=True,
    memory_key="messages",
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

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input("> ")
    result = chain({"content": content})
    print(result["text"])
