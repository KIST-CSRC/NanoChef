@echo off
echo Installing Python packages one by one (including Git-based)...
setlocal enabledelayedexpansion

REM -- 일반 PyPI 패키지 목록 --
set PACKAGES=^
absl-py==2.0.0 ^
astor==0.8.1 ^
astunparse==1.6.3 ^
attrs==23.1.0 ^
backcall==0.2.0 ^
backports.functools-lru-cache==1.6.5 ^
cachetools==5.3.1 ^
certifi==2023.7.22 ^
charset-normalizer==3.3.1 ^
cloudpickle==1.1.1 ^
cma==3.3.0 ^
cycler==0.11.0 ^
Cython==3.0.4 ^
deap==1.4.1 ^
dill==0.3.7 ^
dm-tree==0.1.8 ^
et-xmlfile==1.1.0 ^
exceptiongroup==1.1.3 ^
fasttext==0.9.2 ^
filelock==3.12.2 ^
flatbuffers==23.5.26 ^
fonttools==4.38.0 ^
fsspec==2023.1.0 ^
future==0.18.3 ^
gast==0.2.2 ^
google-auth==2.23.3 ^
google-auth-oauthlib==0.4.6 ^
google-pasta==0.2.0 ^
GPy==1.10.0 ^
GPyOpt==1.2.6 ^
greenlet==3.0.0 ^
grpcio==1.59.0 ^
h5py==3.8.0 ^
huggingface-hub==0.16.4 ^
hyperopt==0.2.7 ^
idna==3.4 ^
importlib==1.0.4 ^
importlib-metadata==6.7.0 ^
iniconfig==2.0.0 ^
ipykernel==5.5.5 ^
ipython==7.33.0 ^
ipython-genutils==0.2.0 ^
jedi==0.19.1 ^
joblib==1.3.2 ^
jupyter-client==5.3.4 ^
jupyter-core==4.5.0 ^
keras==2.11.0 ^
Keras-Applications==1.0.8 ^
Keras-Preprocessing==1.1.2 ^
kiwisolver==1.4.5 ^
libclang==16.0.6 ^
Markdown==3.4.4 ^
MarkupSafe==2.1.3 ^
matplotlib==3.5.3 ^
matplotlib-inline==0.1.6 ^
networkx==2.6.3 ^
numpy==1.21.6 ^
nvidia-cublas-cu11==11.10.3.66 ^
nvidia-cuda-nvrtc-cu11==11.7.99 ^
nvidia-cuda-runtime-cu11==11.7.99 ^
nvidia-cudnn-cu11==8.5.0.96 ^
oauthlib==3.2.2 ^
openpyxl==3.1.2 ^
olymp==0.0.1b0 ^
opt-einsum==3.3.0 ^
packaging==23.2 ^
pandas==1.3.5 ^
paramz==0.9.5 ^
parso==0.8.3 ^
pexpect==4.8.0 ^
phoenics==0.2.0 ^
pickleshare==0.7.5 ^
Pillow==9.5.0 ^
pluggy==1.2.0 ^
prompt-toolkit==3.0.39 ^
protobuf==3.19.6 ^
ptyprocess==0.7.0 ^
py4j==0.10.9.7 ^
pyaml==23.5.8 ^
pyasn1==0.5.0 ^
pyasn1-modules==0.3.0 ^
pybind11==2.11.1 ^
pyDOE==0.3.8 ^
Pygments==2.16.1 ^
pyparsing==3.1.1 ^
pyswarms==1.3.0 ^
pytest==7.4.2 ^
python-dateutil==2.8.2 ^
pytz==2023.3.post1 ^
PyYAML==6.0.1 ^
pyzmq==24.0.1 ^
regex==2023.10.3 ^
requests==2.31.0 ^
requests-oauthlib==1.3.1 ^
rsa==4.9 ^
safetensors==0.4.1 ^
scikit-learn==1.0.2 ^
scikit-optimize==0.10.2 ^
scipy==1.7.3 ^
seaborn==0.12.2 ^
silence-tensorflow==1.2.1 ^
six==1.16.0 ^
sobol-seq==0.2.0 ^
SQCommon==0.3.2 ^
SQLAlchemy==1.4.45 ^
SQSnobFit==0.4.5 ^
support-developer==1.0.5 ^
tensorboard==1.15.0 ^
tensorboard-data-server==0.6.1 ^
tensorboard-plugin-wit==1.8.1 ^
tensorflow==1.15.0 ^
tensorflow-estimator==1.15.1 ^
tensorflow-io-gcs-filesystem==0.34.0 ^
tensorflow-probability==0.8.0 ^
termcolor==2.3.0 ^
threadpoolctl==3.1.0 ^
tokenizers==0.13.3 ^
tomli==2.0.1 ^
torch==1.13.1 ^
torchaudio==0.13.1 ^
torchinfo==1.8.0 ^
torchsummary==1.5.1 ^
torchvision==0.14.1 ^
tornado==6.2 ^
tqdm==4.66.1 ^
traitlets==5.9.0 ^
transformers==4.30.2 ^
typing_extensions==4.7.1 ^
urllib3==2.0.7 ^
watchdog==3.0.0 ^
wcwidth==0.2.8 ^
Werkzeug==2.2.3 ^
wrapt==1.15.0 ^
zipp==3.15.0

REM -- 일반 패키지 설치 루프 --
for %%P in (%PACKAGES%) do (
    echo Installing %%P ...
    pip install %%P || echo Failed to install %%P
)

echo.
echo Installing GitHub-based package: olympus
pip install -e git+https://github.com/aspuru-guzik-group/olympus.git@440b6b58ebfcaa2391cff7e94b570fb4fda98d68#egg=olympus || echo Failed to install olympus from GitHub

echo.
echo ✅ Done installing all packages including GitHub sources.
pause
