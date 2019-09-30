# MEGATHON-QUALCOMM
> Identification of papers/websites/documents with content related to a given context or abstract

## Running the Solution
> make sure the csv files are in the cloned repository
```bash
$ cp $PATH:summaries.csv $PATH:fulltexts.csv .
$ pip3 install -r requirements.txt
$ python3 solution.py
```
> This will create a new file "similarity_matrix.csv" containing a table of similarity between each summary and given text (rows represent the abstracts in summaries.csv, columns represent the texts in fulltexts.csv)
