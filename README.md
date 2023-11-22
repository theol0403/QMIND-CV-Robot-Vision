# QMIND-cv-depthmap

## Setting up the Conda Environment

this only works about half the way through as i havent set up yml file yet. will do soon
To recreate the Conda environment, follow these steps:

1. follow this guide to access caslabs system on vscode https://courses.caslab.queensu.ca/how-to/connect-ssh-sftp-to-caslab-linux-with-visual-studio-code-vs-code/
2. Clone the repository into your caslabs linux enviroment
3. cd into the repository and run this command in your terminal - wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
4. then run - bash Anaconda3-5.2.0-Linux-x86_64.sh
5. install and accept agreement
6. reset vs code
7. return to repository and run - conda env create -n QMIND -f environment.yml
8. activate your enviroment with - conda activate QMIND

now your should be set up to wor on the project


   
