FROM python:3.11-slim

WORKDIR /app

ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD sh -c "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.fileWatcherType=none"
