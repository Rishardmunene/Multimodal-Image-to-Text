# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install torch transformers diffusers pillow pyyaml numpy

# Navigate to your Drive directory where the zip file is located
# Assuming the zip file is in the root of your Google Drive
%cd /content/drive/MyDrive

# Unzip the project
!unzip -o "Multimodal-Image-to-Text.zip" -d "/content/project"

# Navigate to the project directory
%cd /content/project/Multimodal-Image-to-Text

# Verify the contents of the directory
!ls

# Run the main script
!python main.py 