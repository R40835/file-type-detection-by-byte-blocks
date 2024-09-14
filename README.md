# file-type-detection-by-byte-blocks
In this project, we detect file types based on the bytes that constitute them. We use the first, body, and last blocks of bytes on the disk to account for all possible scenarios and train the FFNN, CNN, GRU, and LSTM models. Afterward, we make predictions and evaluate the performance of each model. The experimental computer uses an SSD, where each block size is 4KB, equivalent to 4096 bytes. The selected blocks vary in nature: the first and last blocks may contain headers and trailers for certain file types, whereas the body block presents a greater challenge, as it may lack the distinct patterns often found in the other blocks.

# Dataset
The dataset used for this project consists of files that can be downloaded <a href="https://corp.digitalcorpora.org/corpora/files/govdocs1/zipfiles/">here</a>.
Alternatively, a web-scraping script has been implemented to download the dataset in "toolkit/scrape.py".

# Working dir
- <b>Data visualisation:</b> A dedicated notebook that manages dataset download 
and sampling, creates and analyses visualisations, performs feature extraction, 
and helps interpret data trends, validate assumptions, and communicate 
insights effectively.
- <b>Models random search:</b> A specific notebook designed for hyperparameter 
optimisation using random search, enabling us to efficiently explore a range of 
parameter values and improve model performance.
- <b>Venv:</b> A Python virtual environment used to isolate the dependencies installed 
for the project, useful for avoiding conflicting library versions.
- <b>Requirements:</b> A text file listing the required dependencies to install in the 
project’s virtual environment.
- <b>HPS results:</b> A folder to save the models’ hyperparameter search results for 
each model addressed.
- <b>Toolkit:</b> A python package developed for the project.
- <b>Govdocs1:</b> A folder containing the dataset to be used in the project, consisting 
of files of mixed types.
- <b>Systems 1-6:</b> Separate notebooks that focus on model training and evaluation, 
providing a structured approach to experimenting with different algorithms and 
hyperparameters.

![image](https://github.com/user-attachments/assets/639efb76-cdf4-4137-9376-5043399e7faf)

# Best accuracy scores
![image](https://github.com/user-attachments/assets/f7a08f00-057d-4641-a428-5adfdaee297d)

