# Instacart Analysis

## Install Requirements
To install requirements, run

```bash
pip3 install -r requirements.txt
```

## Data Files
Due to repository restrictions on data sizes, data files were unable to be uploaded. Before running, it is important to download the data and put them into the `data/` folder such that

```
.
├── data
│   ├── aisles.csv
│   ├── departments.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── orders.csv
│   ├── products.csv
│   └── sample_submission.csv
├── README.md
├── requirements.txt
├── run.py
```

## Run
To run the code, just run the file `run.py`

```bash
python3 run.py
```

and a file of the form `submission-*.csv` will be generated.


The submission file to get the score **0.3812513** is file [submission-2017-11-11-01-10-12.csv](/submission-2017-11-11-01-10-12.csv)

## How It Works
To learn how it works, please read the [documentation](/docs/how-it-works.pdf)
