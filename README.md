# Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale

We provide the code, data, and models to replicate the analyses in 'Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale', published at NeurIPS 2025. Please [cite](#paper-citation) if you use this code.

**Quick Links**

* [**Parallel architecture (Parc) Language Models**](https://huggingface.co/collections/jmichaelov/parc-models): Pythia, Mamba, and RWKV language models trained on the same data, with multiple checkpoints and random seeds.
* [**Data**](https://osf.io/4cfve/overview?view_only=71684a677bf240d5a09ec49363f1385d):
    * **Natural Words in Context (NaWoCo) Dataset**: Dataset of words in natural sentence contexts. To use, copy the `stimuli` directory in the OSF repository to this repository.
    * **Outputs**: Outputs of language models, $n$-grams, and fastText models; and language model evaluation scores. To use, copy the `analyses` directory in the OSF repository to this repository.

**Note**: The OpenWebText index is too large to upload; however, we provide the code to create it in this repository.


## Training The Language Models

All code for model training is provided in the `train_models` directory.

To train each model with seed 0, the code can be run as a `slurm` job using the following command-line inputs:

```
sbatch train_model.sh 32 16 mamba 0 owt 10 4000
sbatch train_model.sh 32 16 pythia 0 owt 10 4000
sbatch train_model.sh 1 512 rwkv 0 owt 10 4000
```

Model architectures:

* Mamba: [`state-spaces/mamba-130m-hf`](https://huggingface.co/state-spaces/mamba-130m-hf)

* Pythia: [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m)

* RWKV: [`RWKV/rwkv-4-169m-pile`](https://huggingface.co/RWKV/rwkv-4-169m-pile)

Each model was trained for 4,000 steps on the OpenWebText corpus (made available at [`Skylion007/openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext) on the Hugging Face Dataset Hub), where each step comprised of 512 batches of text 1024 tokens in length (using gradient accumulation to achieve this batch size). This code saves the model every 10 steps. We use steps 10, 20, 40, 80, 160, 320, 640, 1280, 2560, and 4000 in our analyses. Hyperparameters for model training are provided in `train_models/train_model.py`.

Each model was trained using 8GB of RAM memory and 4 cores of one AMD EPYC Processor (7713, 7643, or 7513) and one NVIDIA A100 GPU. Training the 18 models took a total of just under 31 days of compute.

## Creating the OpenWebText infini-gram index

The code for creating the local `infini-gram` index for the OpenWebText dataset is provided in the `openwebtext_index` directory.

To download the dataset from the Hugging Face Dataset Hub, the following command can be run:

```
sbatch download_dataset.sh 
```

We ran this with one core on a AMD EPYC 7643 processor with 16GB RAM, which took just under an hour.

The following command can be run to create the index:

```
sbatch create_index.sh 
```

This took just under 24 hours with 256GB of RAM on 16 cores on one Intel Xeon CPU E5-2698.

## Constructing NaWoCo

We provide the code used to construct the NaWoCo dataset in `dataset_construction`.

This can be replicated by running the `construct_dataset1.py`, `construct_dataset2.py`, and `construct_dataset3.py` sequentially. We ran these on a single node of a Xeon CPU E5-2650 with 64GB of RAM. The main component that impacts time is the rate limit of the `inifini-gram` API. In our case, running all the code took just under 40 hours.

The files created during this process are provided in at this `stimuli` directory. The final version of the dataset is the `final_dataset.tsv`, and `all_stims.stims` is a version of this prepared specifically for use in our language model log-probability pipeline.


## Calculating language model log-probability
The code for calculating language model log-probability is provided in the `run_models` directory.

The experiment can be run with the following command:

```
sbatch run_all_lms.sh
```

We ran this on the same setup as model training, and this took a total of 37 compute days.

Note that this time also includes the time taken to run our models (OWT-Pythia, OWT-Mamba, and OWT-RWKV) on the data, but we have redacted this portion of the code to preserve anonymity.

All outputs for all models are stored in the `output/lm_surprisal` directory. We do not include these files, but provide the merged outputs of all models (as well as $n$-gram log-probabilities and semantic similarity metrics) in the `analyses` directory.

## Calculating $n$-gram log-probability
The code for calculating $n$-gram model log-probability is provided in the `run_models`.

The experiment can be carried out by running `get_ngrams.py`.

We ran this on the same hardware as was used to construct the dataset. Again, time is highly dependent on the `infini-gram` rate limit. In this case, this took roughly 9 compute hours. Files are saved to `output/ngram`.


## Calculating contextual semantic similarity
The code for calculating contextual semantic similarity is provided in `run_models`.

The experiment can be carried out by running `get_word_distance.py`. Note that this requires two `fasttext` models, namely, the `.bin` files for the English [Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html) and [Wikipedia](https://fasttext.cc/docs/en/pretrained-vectors.html) vectors.

We ran this on the same hardware used to calculate $n$-gram log-probability, which took under an hour. Files are saved to `output/distance`.

## Replicating the analyses in the paper
The code for analyses reported in the paper is provided in the `analyses` directory. `combine_data.py` merges all the results into the single `merged_results.tsv` file. We omit this  file to preserve the anonymity of our models.

`calculate_results.py` contains the code for calculating the correlations and running the linear regressions used in our analysis, while `create_plots.Rmd` contains all the code for generating the plots used in the paper. 

These can be run on most systems (with 8GB+ of RAM) in roughly an hour.

## Assets Created
As part of the research carried out for the paper, the assets created were the code provided in this repository (licensed under the Apache 2.0 License) and the 18 Parc language models, also released under the Apache 2.0 License.

## Acceptable Use

The code and dataset provided here are intended for research use only.

The models we created as part of this research (Parc-Pythia, Parc-Mamba, and Parc-RWKV) are intended for research use and generally not suitable for deployment. They are pretrained on a subset of OpenWebText, which is not well-documented, and thus it possible that they are trained on (and may generate) harmful, offensive, or otherwise inappropriate text, especially as they are not fine-tuned in any way. For the same reason, there is no guarantee that they will generate accurate or truthful text. Rather than fine-tuning our models, we instead recommend fine-tuning the original Pythia, Mamba, RWKV models, as they are trained on many times more data.



## Assests Used

### Datasets:

* [**OpenWebText**](http://Skylion007.github.io/OpenWebTextCorpus): CC0 License

* [**FineWeb**](https://openreview.net/forum?id=n6SCkn2QaG): ODC-By License

### Benchmarks:

* [**Paloma**](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760b2d94398aa61468aa3bc11506d9ea-Abstract-Datasets_and_Benchmarks_Track.html) subsets: 

    * **C4, Dolma v1.5, RefinedWeb, mC4-en, Penn TreeBank**: AI2 ImpACT License - Low Risk Artifacts

    * **WikiText-103**: CC BY-SA

    * **RedPajama**: The original RedPajama data licenses, listed on the Hugging Face dataset [repository](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).


* [**ARC**](https://arxiv.org/abs/1803.05457): Unspecified, but released for research purposes.

* [**LAMBADA**](https://zenodo.org/records/2630551): CC BY 4.0

* [**SciQ**](https://aclanthology.org/W17-4413/): CC BY-NC 3.0

* [**SWAG**](https://aclanthology.org/D18-1009/): MIT license

### Pre-existing Models:

* [**Pythia Models**](https://proceedings.mlr.press/v202/biderman23a.html): Apache 2.0 License

* [**PolyPythia Models**](https://openreview.net/forum?id=bmrYu2Ekdz): Apache 2.0 License

* [**Mamba**](https://openreview.net/forum?id=tEYskw1VY2): Apache 2.0 License

* [**RWKV**](https://doi.org/10.18653/v1/2023.findings-emnlp.936): Apache 2.0 License

* [**OWT-GPT2**](https://crfm.stanford.edu/2021/08/26/mistral.html) (the `mistral` open GPT-2 replication models): Apache 2.0 License

* [**Roberta Toxicity Classifier**](https://aclanthology.org/2022.acl-long.469/): OpenRAIL++ License

### Python Software and Packages:

* [`python`](https://docs.python.org/3/license.html): Python Software Foundation License

* [`scipy`](https://www.nature.com/articles/s41592-019-0686-2): BSD 3-Clause License

* [`pandas`](https://doi.org/10.5281/zenodo.3509134): BSD 3-Clause License

* [`numpy`](https://doi.org/10.1038/s41586-020-2649-2): NumPy license

* [`statsmodels`](https://proceedings.scipy.org/articles/Majora-92bf1922-011): BSD 3-Clause 

* [`tqdm`](https://doi.org/10.5281/zenodo.14231923): MIT License and Mozilla Public Licence v. 2.0

* [`sklearn`](https://www.jmlr.org/papers/v12/pedregosa11a.html): BSD 3-Clause License

* [`pytorch`](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html): BSD 3-Clause License

* [`fasttext`](https://doi.org/10.1162/tacl_a_00051): MIT license

* [`transformers`](https://doi.org/10.18653/v1/2020.emnlp-demos.6): Apache 2.0 License

* [`datasets`](https://doi.org/10.18653/v1/2021.emnlp-demo.21): Apache-2.0 license

* [`datatrove`](https://github.com/huggingface/datatrove): Apache-2.0 license

* [`infini-gram`](https://openreview.net/forum?id=u2vAyMeLMm): Apache 2.0 License

* [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness): MIT License


### R Software and Packages:

* [`R`](https://www.R-project.org/): GNU General Public License 2 and GNU General Public License 3

* [`RStudio`](http://www.posit.co/): GNU Affero General Public License v3

* [`tidyverse`](https://doi.org/10.21105/joss.01686): MIT License

* [`colorspace`](https://doi.org/10.1016/j.csda.2008.11.033): BSD 3-Clause License

* [`corrr`](https://CRAN.R-project.org/package=corrr): MIT License

* [`ggh4x`](https://CRAN.R-project.org/package=ggh4x): MIT License

* [`gridExtra`](https://CRAN.R-project.org/package=gridExtra): GPL-2 and GPL-3 

* [`RColorBrewer`](https://CRAN.R-project.org/package=RColorBrewer): Apache License 2.0

* [`viridisLite`](https://sjmgarnier.github.io/viridisLite/): MIT License


## Paper Citation
Please cite the paper if you use this code:
```
@inproceedings{michaelov_levy_bergen_2025_neurips,
 author = {Michaelov, James and Levy, Roger and Bergen, Benjamin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {102407--102447},
 publisher = {Curran Associates, Inc.},
 title = {Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/9432b29b0e25991aa28aff8c7bbe281c-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}

```
