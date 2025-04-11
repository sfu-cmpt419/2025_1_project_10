# SFU CMPT 419 Project FieldView X-Ray: Anatomical Field of View Detection

**XAIVision** (X-ray Anatomical Intelligent Vision System) is a deep learning-based tool for detecting the anatomical field of view from X-ray images.

### Team Members:

- Kai Cho
- Ekamdeep Kaur
- Sung Han
- Iain Chun
- Sanghyeok Park


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/ESRhnQpkI5dKg9GvZc4fUsABxeIMR_tcFHX_5iz8kF9W0Q?e=WvBvzq) |
 [Slack channel](https://cmpt419spring2025.slack.com/archives/C086CRMLGLS) |
 [Project report](https://www.overleaf.com/2253418857zcztgqzwfpgm#37e079) |
 [Database](https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier) |

## Demo Video
[![Watch the demo](https://img.youtube.com/vi/hTeoiOdVxuU/0.jpg)](https://www.youtube.com/watch?v=hTeoiOdVxuU)

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Running the Website with Model Integration](#webs)

5. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

Minimal example to demonstrate website prediction workflow:
```python
# Example: Sending an image to the backend
import requests

files = {'file': open('example_xray.jpg', 'rb')}
response = requests.post('http://127.0.0.1:5000/predict', files=files)
print(response.json())
```

### What to find where


```bash
repository
├── app.py                ## Flask backend server for model inference
├── model_front.py        ## PyTorch model architecture used for webpage(CNNClassifier)
├── download_data.py      ## Script to automatically download the dataset (from Google Drive)
├── main.ipynb            ## Main training notebook (finalized version)
├── main.py               ## .py file of the main training code
├── src/                  ## (Empty or optional scripts folder - not actively used now)
│   └── preprocess.py         ## Preprocessing helper functions
├── LICENSE               ## Open-source license information
├── README.md             ## Project documentation
├── requirements.yml      ## Conda environment specifications
├── xaivision.html        ## Frontend webpage (XAIVision user interface)


```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.
Follow these steps to install the project dependencies and environment:

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate cmpt419_project
```

<a name="repro"></a>
## 3. Reproduction
Steps to reproduce the project setup:
```bash
# Step 1: Download Dataset
python download_data.py  # (Downloads dataset from Google Drive automatically)

# Step 2: Train Model
conda activate cmpt419_project    # (Skip this if already activated)
python model.py

# Step 3: Launch Web Application
python app.py
```
- Data: The dataset will be downloaded to the project root directory automatically.

- Output: Model checkpoints will be saved as xray_model.pth.

<a name="webs"></a>
## 4. Running the Website with Model Integration
After training or loading a pre-trained model ```xray_model.pth```, you can integrate it with the website as follows:
### 1. Make sure these files are present:

- ```app.py``` (Flask backend)

- ```model_front.py``` (CNN model definition)

- ```xray_model.pth``` (trained model weights)

- ```xaivision.html``` (frontend)

### 2. Launch the Flask server:

```bash
conda activate cmpt419_project    # (Skip this if already activated)
python app.py
```
### 3. Open the frontend:

Open ```xaivision.html``` directly in your web browser (double-click or right-click → Open with Browser).

### 4. Test the website:

Upload an X-ray image using the "Upload an X-ray Image" button.

The image will be sent to the backend ```localhost:5000/predict``` and the predicted anatomical field of view will be displayed.

<a name="guide"></a>
## 5. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
