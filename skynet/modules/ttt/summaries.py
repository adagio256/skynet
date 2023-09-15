import timeit
import asyncio
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from skynet.models.v1.job import JobId, JobStatus, JobType
from sklearn.cluster import KMeans
from skynet.models.v1.document import DocumentPayload
from skynet.env import llama_path, llama_n_ctx, llama_n_gpu_layers, llama_n_batch
from skynet.modules.ttt.jobs import create_job, update_job
from skynet.prompts.action_items import action_items_template
from skynet.prompts.summary import summary_template

class SummariesChain:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.llm = LlamaCpp(
            model_path=llama_path,
            temperature=0.01,
            max_tokens=llama_n_ctx,
            n_ctx=llama_n_ctx,
            n_gpu_layers=llama_n_gpu_layers,
            n_batch=llama_n_batch,
        )

        self.__createEmbeddingsForModel()

    def __createEmbeddingsForModel(self) -> None:
        self.embeddings = LlamaCppEmbeddings(
            model_path=llama_path,
            n_ctx=llama_n_ctx,
            n_batch=llama_n_batch,
            n_gpu_layers=llama_n_gpu_layers,
        )

    def __filterByDominantTopics(self, docs: List[Document]) -> List[Document]:
        start = timeit.default_timer()

        topicsCount = np.min([len(docs), 10])

        print(f"Filtering {len(docs)} documents by {topicsCount} topics")

        texts = [doc.page_content for doc in docs]
        vectors = self.embeddings.embed_documents(texts)
        kmeans = KMeans(n_clusters=topicsCount, random_state=42).fit(vectors)
        closest_indices = []

        print(f"KMeans labels: {kmeans.labels_}")
        print(f"KMeans cluster centers: {kmeans.cluster_centers_}")

        for i in range(topicsCount):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_indices.append(np.argmin(distances))

        selected_indices = sorted(closest_indices)
        selected_docs = [docs[i] for i in selected_indices]

        print(f"Selected indices: {selected_indices}")
        print(f"Selected documents: {selected_docs}")

        end = timeit.default_timer()

        print(f"Time to filter documents by topics: {end - start}")

        return selected_docs

    async def __process(self, text: str, template: str, job_id: str) -> str:
        if not text:
            return ""

        loop = asyncio.get_running_loop()
        prompt = PromptTemplate(template=template, input_variables=["text"])
        docs = []
        chain = None
        chunk_size = llama_n_ctx * 3
        create_multiple_docs = len(text) > chunk_size

        if create_multiple_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
            docs = await loop.run_in_executor(
                self.executor,
                self.__filterByDominantTopics,
                text_splitter.create_documents([text])
            )

            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                combine_prompt=prompt,
            )
        else:
            docs = [Document(page_content=text)]

            chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=prompt,
            )

        has_failed = False
        result = None

        try:
            result = await loop.run_in_executor(self.executor, chain.run, docs)
        except Exception as e:
            has_failed = True
            result = str(e)

        update_job(
            job_id,
            status=JobStatus.ERROR if has_failed else JobStatus.SUCCESS,
            result=result
        )

    async def start_summary_job(self, payload: DocumentPayload) -> JobId:
        job_id = create_job(job_type=JobType.SUMMARY)

        task = self.__process(payload.text, template=summary_template, job_id=job_id)
        asyncio.create_task(task)

        return JobId(id=job_id)

    async def start_action_items_job(self, payload: DocumentPayload) -> JobId:
        job_id = create_job(job_type=JobType.ACTION_ITEMS)

        task = self.__process(payload.text, template=action_items_template, job_id=job_id)
        asyncio.create_task(task)

        return JobId(id=job_id)
