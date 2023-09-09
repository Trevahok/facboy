import json 
import os 
from tqdm.auto import tqdm
from uuid import uuid4

from dotenv import load_dotenv
import pinecone

from lib import init_pinecone, embedder, chunkifier


load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX")

def load_dataset():
    file = open('./data/scraped_data.json', 'r')
    dataset = json.load(file)
    file.close()
    return dataset


if __name__ == '__main__':

    print("SETTING UP PINECONE...")
    init_pinecone()

    print("LOADING DATA...")
    dataset = load_dataset()
    embed = embedder()
    chunkifier = chunkifier()
    index = pinecone.GRPCIndex(PINECONE_INDEX)

    print('PINECONE INDEX: ', index.describe_index_stats()  )



    batch_limit = 100

    texts = []
    metadatas = []

    for i, row in enumerate(tqdm(dataset)):
        if row['data'] in ['', None] :
            print("SKIPPING : ", row['url'] )
        print('PROCESSING: ', row['url'])
        metadata = {
            'source': row['url'],
        }
        # now we create chunks from the record text
        record_texts = chunkifier.split_text(row['data'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))


    print("COMPLETED SUCCESSFULLY...")
    print(index.describe_index_stats())






