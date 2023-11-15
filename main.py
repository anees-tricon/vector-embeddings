import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import VectorDBQA, OpenAI

pinecone.init(

    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="gcp-starter",
)

if __name__ == "__main__":
    print("Hello Embeddings")

    loader = TextLoader(
        "/Users/anees/Code/llm/langchain-tutorials/embeddings/mediumblog1.txt"
    )

    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blog-index"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )
    query = "What are some of the ethical considerations that must be taken into account when implementing advanced vector embeddings in data analysis?"
    result = qa({"query": query})
    print(result)

    # pinecone.delete_index("medium-blog-index")
