FROM python:3.10
WORKDIR /Adv-ML-Second-Module
COPY . /Adv-ML-Second-Module
RUN pip install ./dist/adv_ml_first_module-0.1.0-py3-none-any.whl
CMD ["python", "Adv-ML-Second-Module/model.py"]