# Mobiquity
This project aims to streamline data processing operations for multiple projects in the Urban Mobility, Networks and Intelligence (UMNI) lab at Purdue University, led by [Dr. Satish Ukkusuri](http://www.satishukkusuri.com/). The main modules involve analyzing '**mobi**lity' and 'transport e**quity**'.

## Files and resources:
<!-- - **[Framework in Figma](https://www.figma.com/file/LqnQC54G4w6CaDwsGZExXU/Mobil?node-id=0%3A1&t=kH061lIHBTjiACSy-1)** -->
- **Mobiquity.pptx**: Main presentation to record our weekly/biweekly updates.
- **[Notion document](https://emphasent.notion.site/Mobilkit-aa39edb3dd77487aac1768671a3a75ee)**: For documenting ideas and content details such as codebase description.

## Installation
The current version of `mobiquity` uses `pyspark` for which it requires Python 3 version 3.9 or earlier.
It is recommended to install this package in a new virtual environment. In `conda`, this may be done as:
```bash
conda create -n mq python=3.9.7
conda activate mq
```
Then, it can be installed using `pip` from [PyPi](https://pypi.org/project/pip/):
```bash
pip install git+https://rvanxer@github.com/rvanxer/mobiquity.git
```
