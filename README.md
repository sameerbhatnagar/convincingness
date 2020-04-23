# How to reproduce results (in progress)

```
$ virtualenv -p /usr/bin/python3.7 venv_convincingness
$ source venv_convincingness/bin/activate

(venv_convincingness) $ pip install pip-tools
(venv_convincingness) $ pip-sync requirements/requirements.txt
(venv_convincingness) $ cd code
(venv_convincingness) $ jupyter notebook BERT_Fine_Tuning.ipynb
```
The above Bert Fine tuning needs to have access to a GPU. The results will be placed in a directory called `tmp`.

To run all other models and reproduce graphs in article:
```
(venv_convincingness) $ jupyter notebook convincingness.ipynb
```
