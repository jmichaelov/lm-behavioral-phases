from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor

pipeline = [
    HuggingFaceDatasetReader("Skylion007/openwebtext", dataset_options={"split":"train"},streaming=True),
    JsonlWriter(output_folder="full_owt")
]

def run():
    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline, workers=16, tasks=16)
    print(executor_1.run())


if __name__ == "__main__":
    run()