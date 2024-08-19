# DLCRec

This is the code for DLCRec (DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems) for WSDM 2025, a novel framework enables LLMs to output recommendations with varying user requests for diversity.

![FIg1](./figures/framework.png "Framework")

## Data
We experiment on two real-world datasets, [MovieLens-10M](https://grouplens.org/datasets/movielens/10m/) and [Steam](https://github.com/kang205/SASRec). Download the data and run the jupyter notebook in "./data" to generate task-specific data and the augmented data. 

## Environment

We use the same environment as [Llama Factory](https://github.com/hiyouga/LLaMA-Factory). You can also use the following code to create a conda environment:

```
conda env create -f llama_factory.yaml
```



## Training framework

DLCRec decompose the diversity-oriented controllable recommendation into 3 specific sub-tasks: genre predicting (GP), genre filling (GF) and item predicting (IP). 

To train each model, select the training and validation set and run "./train.sh".

To infer each model, run
```
python inference.py --test_data_path ./data/movie_sgcate/test_1000_nonum.json --base_model ../Meta-Llama-3-8B-Instruct/ --lora_weights ./model/movie_sgcate/ModelName  --result_json_data ./movie_sgcate_result/ModelName_GP1000gred.json --num_beams 1 --num_beam_groups 1 --do_sample False --batch_size 64
```
```
python inference.py --test_data_path ./data/movie_sgcate/test_1000_BERT_GF.json --base_model ../Meta-Llama-3-8B-Instruct/ --lora_weights ./model/movie_sgcate/ModelName  --result_json_data ./movie_sgcate_result/ModelName_GF1000gred.json --num_beams 1 --num_beam_groups 1 --do_sample False --batch_size 64
```
```
python inference.py --test_data_path ./data/movie_sgcate/test_1000_BERT_IP.json --base_model ../Meta-Llama-3-8B-Instruct/ --lora_weights ./model/movie_sgcate/ModelName  --result_json_data ./movie_sgcate_result/ModelName_IP1000gred.json --num_beams 1 --num_beam_groups 1 --do_sample False --batch_size 16
```
To generate the embeddings of items for Grounding, run
```
python ./data/movie_sgcate/generate_embedding.py
```
To evaluate each model, run
```
python ./data/movie_sgcate/evaluate_GP.py
```
```
python ./data/movie_sgcate/evaluate_GF.py
```
```
python ./data/movie_sgcate/evaluate_IP.py
```
## Control framework
The user can select the control number of genres as they wish, and DLCRec propagate the control signals through 3 sub-tasks to form the final recommendation. We do the following steps to operate the control framework:
+ First, we infer the model of task GP to get the predicted genres given the control number which is 5 here:
    ```
    python inference_controlGPnum.py --test_data_path ./data/movie_sgcate/test_1000_GP.json --base_model ../Meta-Llama-3-8B-Instruct/ --lora_weights ./model/movie_sgcate/GPModelName  --result_json_data ./movie_sgcate_result/GPModelName_5GP1000gred.json --num_beams 1 --num_beam_groups 1 --do_sample False --batch_size 64 --control_GPnum 5
    ```
+ Second, we transfer format of task GP's output into the format of task GF's input. We put the result file path into the file "./data/movie_sgcate/modifyGF_controlGP.py" and then run:
    ```
    python ./data/movie_sgcate/modifyGF_controlGP.py
    ```
    witch put the control target into the file "./data/movie_sgcate/test_1000_BERT_GF_controlGP.json".

+ Third, we infer the model of task GF to get the genres of future items given the control target by task GP. We put the control number 5 in the file path to facilitate evaluation.
    ```
    python inference_controlGP.py --test_data_path ./data/movie_sgcate/test_1000_BERT_GF_controlGP.json --base_model ../Meta-Llama-3-8B-Instruct/ --lora_weights ./model/movie_sgcate/GFModelName  --result_json_data ./movie_sgcate_result/GFModelName_controlControlTarget_5GF1000gred.json --num_beams 1 --num_beam_groups 1 --do_sample False --batch_size 64 --control_GP ControlTarget
    ```
+ Fourth, we transform format of task GF's output into the format of task IP's input. We put the result file path into the file "./data/movie_sgcate/modifyIP_controlGF.py" and then run:
    ```
    python ./data/movie_sgcate/modifyIP_controlGF.py
    ```
+ Finally, we infer the model of task IP to get the future genres given the control target by task GF. We put the control number 5 in the file path to facilitate evaluation.
    ```
    python inference_controlGF.py --test_data_path ./data/movie_sgcate/test_1000_BERT_IP_controlGF.json --base_model ../Meta-Llama-3-8B-Instruct/ --lora_weights ./model/movie_sgcate/IPModelName  --result_json_data ./movie_sgcate_result/IPModelName_controlControlTarget_5GF1000gred_IP1000gred.json --num_beams 1 --num_beam_groups 1 --do_sample False --batch_size 16 --control_GF ControlTarget
    ```

To evaluate each model, also run
```
python ./data/movie_sgcate/evaluate_GP.py
```
```
python ./data/movie_sgcate/evaluate_GF.py
```
```
python ./data/movie_sgcate/evaluate_IP.py
```

## Other baselines：
For BIGRec or BIGRec_CoT or BIGRec_div, the training and infering process are similar as above. To evaluate it, run
```
python ./data/movie_sgcate/evaluate_BIGRec.py
```
For Raw LLM, given to their unstable output format, evaluate it by running:
```
python ./data/movie_sgcate/evaluate_RAW.py
```
