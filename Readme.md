# PRAETORIAN

Repository for **[Trapping Attacker in Dilemma: Examining Internal Correlations and External Influences of Trigger for Defending GNN Backdoors](https://openreview.net/forum?id=7xWSWpsU5J)** (NeurIPS 2026).

## Release status

We are preparing the full open-source release, including updated code and documentation, and expect to publish it between **December 2026 and January 2027**. The files currently in this repository are an early review version. Updates will be published here.

## Quick start

To run the toy example on the Cora dataset under the DPGBA attack:

```bash
bash cora_defense.sh
```

## Reproducibility notes

- The paper uses public graph benchmark datasets and previously published graph
  backdoor attack/defense protocols.
- Dataset loaders and preprocessing scripts are included where applicable.
- If a dataset is not bundled locally, the corresponding script downloads it
  from the original public source or from the standard benchmark loader.
- This repository does not claim ownership over any third-party dataset,
  library, or implementation.

## Third-party datasets and license / terms status

The following public datasets are used only for academic benchmarking and
reproducibility. Please cite the original dataset papers/sources when using this
repository.

| Dataset | Source used in this package | Citation / credit | License or terms status |
|---|---|---|---|
| Cora | LINQS citation-network benchmark, also available through common GNN dataset loaders | McCallum et al., *Automating the Construction of Internet Portals with Machine Learning*, 2000 | The original LINQS release is publicly available, but we did not find an explicit dataset license in the original release metadata. We use it as a standard academic benchmark and cite the original source. |
| PubMed / PubMed Diabetes | LINQS PubMed Diabetes citation-network benchmark | Sen et al., *Collective Classification in Network Data*, 2008 | The original LINQS release is publicly available, but we did not find an explicit dataset license in the original release metadata. We use it as a standard academic benchmark and cite the original source. |
| OGB-arxiv / ogbn-arxiv | Open Graph Benchmark (OGB) node-property prediction dataset | Hu et al., *Open Graph Benchmark: Datasets for Machine Learning on Graphs*, 2021 | OGB lists `ogbn-arxiv` under the Open Data Commons Attribution License (ODC-BY). Users should follow the OGB terms and attribution requirements. |
| Flickr | Flickr node-classification benchmark from GraphSAINT, available through DGL/PyG dataset loaders | Zeng et al., *GraphSAINT: Graph Sampling Based Inductive Learning Method*, 2020 | The benchmark is distributed through the GraphSAINT/DGL/PyG ecosystem. The GraphSAINT repository is MIT licensed. The benchmark uses processed graph features and labels rather than raw Flickr images; original Flickr content remains subject to the licenses chosen by the original Flickr content owners. |

## Third-party code and libraries

This package may use the following public research software and libraries:

- DGL for graph construction and message passing.
- PyTorch and standard scientific Python packages for model training and
  evaluation.
- OGB for standardized loading and evaluation of `ogbn-arxiv`.
- Public implementations or protocols of graph backdoor attacks and defenses,
  cited in the paper where used.

All third-party code and libraries remain under their own licenses. Users should
consult the corresponding upstream repositories for exact license files,
version-specific terms, and dependency requirements.

## License

See [LICENSE](LICENSE) for the repository license. Third-party datasets, libraries, and implementations remain under their original licenses and terms, as summarized above.

## Dataset handling policy

- We use public benchmark data only for research and reproducibility.
- We do not claim ownership of third-party datasets.
- We do not use private, scraped, or human-subject data collected by the authors.
- For datasets whose original release does not clearly state a license, we
  report that status explicitly and rely on the original public benchmark source
  and academic citation practice.
- If redistribution terms are unclear for a deployment or downstream use case,
  users should download the dataset directly from the original source and comply
  with the original provider's terms.

