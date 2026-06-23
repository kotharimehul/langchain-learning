from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

import os
def main():
    file_path = Path(__file__).parent / "mediumblog.txt"
    loader = TextLoader(str(file_path), encoding="utf-8")

    document = loader.load()
    print(f"Loaded {len(document)} document(s), {len(document[0].page_content)} characters")

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024,
    )
    print("Ingesting")

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
    )
    print("finish")

if __name__ == '__main__':
    main()
        


