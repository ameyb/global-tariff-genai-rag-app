### Installation


### Install Mesa OpenGL library in ubuntu

```
sudo apt-get install libgl1-mesa-glx
sudo apt install poppler-utils
sudo apt install tesseract-ocr
```

### Mac and Linux commands

```
brew install tesseract
brew install poppler
brew install postgresql

pip3 install psycopg2-yugabytedb
pip3 install psycopg2-yugabytedb-binary
pip3 install yb-cassandra-driver
pip3 install langchain langchain-community python_dotenv
pip3 install langchain-openai

pip3 install pandas numpy
pip3 install streamlit

pip3 install "unstructured[all-docs]<=0.16.10"
pip3 install langchain_postgres

pip3 uninstall pdfminer.six
pip3 install pdfminer.six==20221105
pip3 install ipython
```

### run the app

```
export OPENAI_API_KEY=
export PGDATABASE=yugabyte 
streamlit run RAG_with_streamlit.py 
```

### Start the YugabyteDB Cluster

```
https://docs.yugabyte.com/preview/releases/ybdb-releases/v2.25/ 


./bin/yugabyted start --base_dir /tmp/node1 --enable_pg_parity_early_access \
--master_flags="ysql_yb_enable_advisory_locks=true,allowed_preview_flags_csv={ysql_yb_enable_advisory_locks}" \
--tserver_flags="ysql_yb_enable_advisory_locks=true,allowed_preview_flags_csv={ysql_yb_enable_advisory_locks}" \
--master_webserver_port=9999
```