# gig_loan_approval_predictor

conda create -n gigloan python=3.8 -y

conda activate gigloan

python setup.py install

pip install -r requirements.txt

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

docker pull prom/prometheus

docker run -d -p 9090:9090 -v /c/test/gig_loan_approval_predictor/prometheus.yaml:/etc/prometheus/prometheus.yml:ro prom/prometheus
http://localhost:9090

docker pull grafana/grafana

docker run -d -p 3000:3000 grafana/grafana
http://localhost:3000

docker rm $(docker ps -aq)
docker build -t gigloan_image .
docker-compose down
docker-compose up -d

python3 -m venv mlflow-env
source mlflow-env/bin/activate
pip install --upgrade pip
pip install mlflow
nohup mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
tail -f mlflow.log

