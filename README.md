# rag-test



## Install dependenies
1. From requirements file
```
pip install -r requirements.txt
```

2. Install markdown dependenies
```
pip install "unstructured[md]"
```

## Test

1. Create database
```
python create_database.py
```

2. Test query 
```
python app.py "How does Alice meet the Mad Hatter?"
```