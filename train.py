import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


# In-Memory Document Store
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

#-----------------------------------------------------------------------------------------------
from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor

converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["jp"])
doc = converter.convert(file_path="説明書.pdf", meta=None)[0]

docs = [doc]
# Let's have a look at the first 3 entries:
print(docs[:3])

# Now, let's write the docs to our DB.
document_store.write_documents(docs)

#-----------------------------------------------------------------------------------------------
# An in-memory TfidfRetriever based on Pandas dataframes
from haystack.nodes import TfidfRetriever

retriever = TfidfRetriever(document_store=document_store)


#-----------------------------------------------------------------------------------------------

from haystack.nodes import FARMReader


# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)



#-----------------------------------------------------------------------------------------------
from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)

#-----------------------------------------------------------------------------------------------
# You can configure how many candidates the reader and retriever shall return
# The higher top_k for retriever, the better (but also the slower) your answers.
prediction = pipe.run(
    query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

import pickle

pickle.dump(model, open("model.pickle", 'wb'))