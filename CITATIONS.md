# Citations & Acknowledgments

This document provides complete citations for all datasets and models used in stick-gen.

## AIST Dance Video Database

```bibtex
@inproceedings{aist-dance-db,
  author = {Shuhei Tsuchida and Satoru Fukayama and Masahiro Hamasaki and Masataka Goto}, 
  title = {AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing}, 
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference, {ISMIR} 2019},
  address = {Delft, Netherlands}, 
  year = 2019, 
  month = nov }
```

## 100STYLE Dataset

The stick-gen project uses the 100STYLE dataset for training realistic human motion with camera calibration.

```bibtex
@dataset{mason_2022_8127870,
  author       = {Mason, Ian and
                  Starke, Sebastian and
                  Komura, Taku},
  title        = {100STYLE Dataset},
  month        = may,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.1145/3522618},
  url          = {https://doi.org/10.1145/3522618},
}
```

## AMASS Dataset

The stick-gen project uses the AMASS (Archive of Motion Capture as Surface Shapes) dataset for training realistic human motion.

### Main AMASS Citation

```bibtex
@inproceedings{AMASS:ICCV:2019,
  title = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  pages = {5442--5451},
  month = oct,
  year = {2019}
}
```

## SMPL+H Body Model

The AMASS dataset uses the SMPL+H body model representation.

```bibtex
@article{MANO:SIGGRAPHASIA:2017,
  title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
  author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  pages = {245:1--245:17},
  month = nov,
  year = {2017}
}
```

## SMPL+H Compatible Datasets Used

Stick-gen uses 12 SMPL+H compatible datasets from AMASS (5,592 motion sequences total):

### 1. CMU Motion Capture Database

```bibtex
@misc{AMASS_CMU,
  title = {{CMU MoCap Dataset}},
  author = {{Carnegie Mellon University}},
  url = {http://mocap.cs.cmu.edu}
}
```

### 2. MPI_Limits

```bibtex
@inproceedings{AMASS_PosePrior,
  title = {Pose-Conditioned Joint Angle Limits for {3D} Human Pose Reconstruction},
  author = {Akhter, Ijaz and Black, Michael J.},
  booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2015}
}
```

### 3. TotalCapture

```bibtex
@inproceedings{AMASS_TotalCapture,
  author = {Trumble, Matt and Gilbert, Andrew and Malleson, Charles and Hilton, Adrian and Collomosse, John},
  title = {{Total Capture}: 3D Human Pose Estimation Fusing Video and Inertial Sensors},
  booktitle = {2017 British Machine Vision Conference (BMVC)},
  year = {2017}
}
```

### 4. Eyes_Japan_Dataset

```bibtex
@misc{AMASS_EyesJapanDataset,
  title = {{Eyes Japan MoCap Dataset}},
  author = {Eyes JAPAN Co. Ltd.},
  url = {http://mocapdata.com}
}
```

### 5. KIT Motion Capture Database

```bibtex
@inproceedings{AMASS_KIT,
  author = {Christian Mandery and \"Omer Terlemez and Martin Do and Nikolaus Vahrenkamp and Tamim Asfour},
  title = {The {KIT} Whole-Body Human Motion Database},
  booktitle = {International Conference on Advanced Robotics (ICAR)},
  pages = {329--336},
  year = {2015}
}

@article{AMASS_KIT_2,
  author = {Christian Mandery and \"Omer Terlemez and Martin Do and Nikolaus Vahrenkamp and Tamim Asfour},
  title = {Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion},
  pages = {796--809},
  volume = {32},
  number = {4},
  journal = {IEEE Transactions on Robotics},
  year = {2016}
}
```

### 6. BioMotionLab_NTroje

```bibtex
@article{AMASS_BMLrub,
  title = {Decomposing Biological Motion: {A} Framework for Analysis and Synthesis of Human Gait Patterns},
  author = {Troje, Nikolaus F.},
  year = 2002,
  month = sep,
  journal = {Journal of Vision},
  volume = 2,
  number = 5,
  pages = {2--2},
  doi = {10.1167/2.5.2}
}
```

### 7. BMLmovi

```bibtex
@article{AMASS_BMLmovi,
  title = {{MoVi}: A Large Multipurpose Motion and Video Dataset},
  author = {Saeed Ghorbani and Kimia Mahdaviani and Anne Thaler and Konrad Kording and Douglas James Cook and Gunnar Blohm and Nikolaus F. Troje},
  year = {2020},
  journal = {arXiv preprint arXiv: 2003.01888}
}
```

### 8. EKUT

Part of the KIT Motion Capture Database (see KIT citation above).

### 9. TCD_handMocap

```bibtex
@inproceedings{AMASS_TCDHands,
  author = {Ludovic Hoyet and Kenneth Ryall and Rachel McDonnell and Carol O'Sullivan},
  title = {Sleight of Hand: Perception of Finger Motion from Reduced Marker Sets},
  booktitle = {Proceedings of the ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games},
  year = {2012},
  pages = {79--86},
  doi = {10.1145/2159616.2159629}
}
```

### 10. ACCAD

```bibtex
@misc{AMASS_ACCAD,
  title = {{ACCAD MoCap Dataset}},
  author = {{Advanced Computing Center for the Arts and Design}},
  url = {https://accad.osu.edu/research/motion-lab/mocap-system-and-data}
}
```

### 11. HumanEva

```bibtex
@article{AMASS_HumanEva,
  title = {{HumanEva}: Synchronized video and motion capture dataset and baseline algorithm for evaluation of articulated human motion},
  author = {Sigal, L. and Balan, A. and Black, M. J.},
  journal = {International Journal of Computer Vision},
  volume = {87},
  number = {1},
  pages = {4--27},
  publisher = {Springer Netherlands},
  month = mar,
  year = {2010}
}
```

### 12. MPI_mosh

```bibtex
@article{AMASS_MoSh,
  title = {{MoSh}: Motion and Shape Capture from Sparse Markers},
  author = {Loper, Matthew M. and Mahmood, Naureen and Black, Michael J.},
  address = {New York, NY, USA},
  publisher = {ACM},
  month = nov,
  number = {6},
  volume = {33},
  pages = {220:1--220:13},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  url = {http://doi.acm.org/10.1145/2661229.2661273},
  year = {2014},
  doi = {10.1145/2661229.2661273}
}
```

## Additional Datasets and Licenses

This section lists additional datasets present under the local `data/` directory
that complement the AMASS and 100STYLE resources above. For each dataset we
include a short description, a recommended citation, and a brief license
summary. Always refer to the official dataset website or bundled LICENSE/README
files for the most up-to-date and legally binding terms.

### AIST++ 3D Dance Dataset (camera-aware, stylized dance)

The `data/aist_plusplus` folder contains the AIST++ 3D dance dataset with
multi-view camera parameters, 2D/3D keypoints, and motion sequences. AIST++ is
introduced in the AI Choreographer paper:

```bibtex
@inproceedings{li2021aistpp,
  title   = {{AI Choreographer}: Music Conditioned 3D Dance Generation with {AIST++}},
  author  = {Li, Wen and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year    = {2021}
}
```

License: research-oriented; see the AIST++ project page and the underlying
AIST Dance Video Database license for detailed terms and camera/video usage
restrictions.

### LSMB19 Long-Horizon Motion Benchmark (continuous skeleton sequences)

The `data/lsmb19-mocap` folder contains the LSMB19 benchmark, consisting of two
very long and continuous unsegmented 3D skeleton sequences, cross-subject and
cross-view training/testing splits, query sets, and annotations.

```bibtex
@misc{lsmb19_dataset,
  title        = {{LSMB19}: Dataset for benchmarking search and annotation},
  howpublished = {DISA Laboratory, Masaryk University. \url{https://disa.fi.muni.cz/research-directions/motion-data/data/}},
  year         = {2019}
}
```

License: research use only; see the LSMB19 benchmark homepage for detailed
terms and conditions.

### NTU RGB+D 60 and NTU RGB+D 120 (skeleton-based action datasets)

The `data/NTU_RGB_D` folder contains skeleton-based action recognition data
from the NTU RGB+D 60 and NTU RGB+D 120 datasets.

```bibtex
@inproceedings{shahroudy2016ntu,
  title     = {{NTU RGB+D}: A Large Scale Dataset for 3D Human Activity Analysis},
  author    = {Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@article{liu2019ntu120,
  title   = {{NTU RGB+D 120}: A Large-Scale Benchmark for 3D Human Activity Understanding},
  author  = {Liu, Jun and others},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2019}
}
```

License: non-commercial research only; requires agreement to the NTU RGB+D
dataset license terms from Nanyang Technological University.

### InterHuman / InterGen Multi-Human Interaction Dataset

The `data/InterHuman Dataset` folder contains the InterHuman dataset used by
the InterGen paper for multi-human motion generation under complex
interactions. The official LICENSE in this folder describes a strictly
non-commercial, non-redistributable research license.

```bibtex
@article{liang2024intergen,
  title   = {Intergen: Diffusion-based multi-human motion generation under complex interactions},
  author  = {Liang, Han and Zhang, Wenqian and Li, Wenxuan and Yu, Jingyi and Xu, Lan},
  journal = {International Journal of Computer Vision},
  pages   = {1--21},
  year    = {2024},
  publisher = {Springer}
}
```

License: "Dataset Copyright License for non-commercial scientific research
purposes" (see `data/InterHuman Dataset/LICENSE.md`); non-commercial research
only, no redistribution, and additional content restrictions.

### HumanML3D Text-Motion Dataset

The `data/HumanML3D` folder contains the HumanML3D text-motion dataset and
associated code. HumanML3D provides diverse single-character motions paired
with natural language descriptions.

```bibtex
@inproceedings{guo2022humanml3d,
  title     = {Generating Diverse and Natural 3D Human Motions from Text},
  author    = {Guo, Chuan and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

HumanML3D aggregates and builds on several underlying motion datasets,
including AMASS, KIT Motion-Language (KIT-ML), HumanAct12, and others. AMASS
and its components are cited above. KIT-ML and HumanAct12 are cited below.

License: research use; see the HumanML3D repository README and LICENSE for
precise redistribution and usage terms. Note that underlying datasets retain
their own licenses.

### KIT Motion-Language (KIT-ML) Dataset

The `data/KIT-ML` folder contains the KIT Motion-Language dataset, which
associates motion clips with natural-language descriptions.

```bibtex
@article{Plappert2016,
  author  = {Plappert, Matthias and Mandery, Christian and Asfour, Tamim},
  title   = {The KIT Motion-Language Dataset},
  journal = {Big Data},
  volume  = {4},
  number  = {4},
  pages   = {236--252},
  year    = {2016}
}
```

License: research/non-commercial; see the KIT Motion-Language dataset website
for current license details and terms of use.

### Category Overview

For quick reference, the main datasets and their primary roles in stick-gen are:

- **Long-horizon motion / continuous sequences:** LSMB19, AMASS (see above).
- **Skeleton-based action recognition:** NTU RGB+D 60/120.
- **Dance / stylized motion with cameras:** AIST Dance Video Database,
  AIST++, 100STYLE.
- **Text-conditioned motion:** HumanML3D, KIT-ML (plus AMASS-based subsets).
- **Multi-human interaction:** InterHuman / InterGen.

## Text Embedding Model

Stick-gen uses BAAI/bge-large-en-v1.5 for text embeddings (Top-5 on MTEB leaderboard).

```bibtex
@misc{bge_embedding,
  title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
  author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
  year={2023},
  eprint={2309.07597},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Acknowledgments

We thank:
- The AMASS team at the Max Planck Institute for Intelligent Systems for creating and maintaining the AMASS dataset
- All contributing motion capture labs and institutions for making their data publicly available
- The Beijing Academy of Artificial Intelligence (BAAI) for the BGE embedding model
- The open-source community for PyTorch, Transformers, and related tools

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The AMASS dataset and individual motion capture datasets have their own licenses. Please refer to the [AMASS website](https://amass.is.tue.mpg.de/) for licensing information.

