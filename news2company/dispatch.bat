@echo off
for %%i in (54.238.228.15,13.114.245.76,54.249.45.135,13.231.175.161,3.112.58.223,13.115.59.205,13.230.95.98,52.198.217.93,54.248.205.213,18.182.34.121,13.114.109.17,13.114.53.110,13.231.196.248,13.231.197.55,13.231.249.125,13.115.251.233) do (
echo %%i
ssh -i miodata-key-tk.pem hadoop@%%i "sudo rm -rf /tmp/ner_model;mkdir /tmp/ner_model; cd /tmp/ner_model; aws s3 cp s3://com.miotech.data.prd/Project/miotech/miotech-cn-nlp/news2company/model/model.zip .;unzip model.zip;sudo rm train.log model.zip"
)
pause
