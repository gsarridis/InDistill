# InDistill: Transferring Knowledge From Pruned Intermediate Layers

This is a pytorch implementation of the InDistill method. Our paper can be found **here** (TODO: add the link)

InDistill enchances the effectiveness of the Knowledge Distillation procedure by leveraging the properties of channel pruning to both reduce the capacity gap between the models and retain the information geometry. Also, this method introduces a curriculum learning based scheme for enhancing the effectiveness of transferring knowledge from multiple intermediate layers.

## Python environment

python = 3.6
pytorch = 1.10.1

We also provide the .yml file of the environmnent we used (pytorch36_environment.yml).
You can create an identical environment by running
````
conda env create -f pytorch36_environment.yml
````
## Run the code
First, we train the baseline models by running the corresponding bash file
````
sh ./train_baselines.sh
````
Note that the pretrained models are provided in dir: results/models and you can use them instead of repeating the baseline training procedure.

Then, we transfer the knowledge from the teacher to the auxiliary model by running
````
sh ./train_aux.sh
````
and we prune it 
````
sh ./prune.sh
````
Again, the output models are provided in "results/models".

Finally, we transfer the knowledge from the auxiliary to the student using the following bash file
````
sh ./train_student_from_aux.sh
````
The results can be printed by running
````
python results.py
````

# Cite
If you find this code useful in your research, please consider citing:
```
@article{sarridis2021indistill,
  title={InDistill: Transferring Knowledge From Pruned Intermediate Layers},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  journal={arXiv preprint arXiv:TODO},
  year={2022}
}
```
