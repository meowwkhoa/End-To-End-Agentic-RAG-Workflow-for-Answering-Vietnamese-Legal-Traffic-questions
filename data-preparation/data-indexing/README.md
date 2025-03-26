
conda create -n Data_Ingestion_Pipeline python=3.9

conda activate Data_Ingestion_Pipeline

pip install -r requirements.txt

jupyter lab

kubens -> weaviate
kubectl port-forward svc/weaviate 8085:85
