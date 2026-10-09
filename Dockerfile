FROM python:3.10-slim


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

COPY .streamlit /root/.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
