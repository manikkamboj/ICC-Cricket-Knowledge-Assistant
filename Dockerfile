FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run RAG_Chatbot.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]