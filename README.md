# SHOWMe: Benchmarking Object-agnostic Hand-Object 3D Reconstruction (ICCV / ACVR 2023)

[[Paper](https://europe.naverlabs.com/research/showme/)] [[Project Page](https://europe.naverlabs.com/research/showme)]
<!-- [[Oral Presentation](https://es.naverlabs.com/Humans-NLE/SHOWMe)]  -->
> [Anilkumar Swamy](https://europe.naverlabs.com/people_user/anilkumar-swamy/),
> [Vincent Lerory](https://europe.naverlabs.com/people_user/vincent-leroy/),
> [Philippe Weinzaepfel](https://europe.naverlabs.com/people_user/philippe-weinzaepfel/),
> [Fabien Baradel](https://fabienbaradel.github.io/),
> [Salma Galaaoui](https://europe.naverlabs.com/people_user/salma-galaaoui/),
> [Romain Brégier](https://europe.naverlabs.com/people_user/romain-bregier/),
> [Matthieu Armando](https://europe.naverlabs.com/people_user/matthieu-armando/),
> [Jean-Sebastien Franco](https://morpheo.inrialpes.fr/~franco/),
> [Grégory Rogez](https://europe.naverlabs.com/people_user/gregory-rogez/)       
> *ACVR workshop at ICCV 2023*

This repository contains the link for downloading and code for visualizing SHOWMe Hand-Object dataset.

## Citation
```bibtex
@inproceedings{showme,
  title={{SHOWMe: Benchmarking Object-agnostic Hand-Object 3D Reconstruction}},
  author={{Swamy, Anilkumar and Leroy, Vincent and Weinzaepfel, Philippe and Baradel, Fabien and Galaaoui, Salma and Brégier, Romain and Armando, Matthieu and Franco, Jean-Sebastien and Rogez, Grégory}},
  booktitle={{ICCVW}},
  year={2023}
}
```

## Dataset Download
### [Link](https://download.europe.naverlabs.com/showme)

## News
- [x] Dataset Release
- [x] Visualization Scripts Release

## Scripts 
- [x] To print all sequence ids 
- [x] To visualize RGB frame 
- [x] To visualize RGBD frame 
- [x] To visualize GT meshes (hand-object and object) 
- [x] To visualize projected meshes 
- [x] To visualize mano mesh 
- [x] To visualize pixel-aligned depth point clouds 
- [x] To render semantic maps (render mano hand and object mesh)

## Running Scripts 

### Install
Our code is running using python3.7 and requires the following packages:
- pytorch-1.7.1+cu110
- pytorch3d-0.3.0
- PIL
- numpy
- trimesh
- matplotlib

### Display all sequence ids
```bash
python scripts/dataset_info.py --datadir <dataset_directory_path> 
```

### Visualize 
```bash
python scripts/vis_showme.py --vis_type <visualization_type> --datadir <dataset_directory_path> --depth_datadir <depth_dataset_directory_path> --seq_id <sequence_id> --frm_no <frame_number>
```
Arguments help:

- ```<vis_type>```: visualization type argument ( ```rgb``` or ```rgbd``` or ```depth``` or ```ho_mesh``` or ```obj_mesh``` or ```mano_mesh``` or ```proj_verts``` or ```pix_algnd_depth```  )

### Render (renders mano hand and object)
```bash
python scripts/render.py --datadir <dataset_directory_path> --outdir <output_dir_tosave_rendered_images> --seq_id <sequence_id>
```

## License

The code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.

A summary of the CC BY-NC-SA 4.0 license is located here:
    https://creativecommons.org/licenses/by-nc-sa/4.0/

The CC BY-NC-SA 4.0 license is located here:
    https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

