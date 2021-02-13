# Temporal Localization of Narrated Actions in Vlogs

This repository contains the dataset and code for our todo_conf paper:
[When did it happen? Duration-informed Temporal Localization of Narrated
Actions in Vlogs](todo_arxiv)

## Task Description
![Example instance](images/annotation_example.jpg)
<p align="center"> Given a video and its transcript, temporally localize the human actions mentioned in the video. </p>

## Dataset Overview
![Example instance](images/model_idea.jpg)
<p align="center">Distinguishing between actions that are narrated by the vlogger but not visible in the video
and actions that are both narrated and visible in the video (underlined), with a highlight on visible actions that represent the same
activity (same color). The arrows represent the temporal alignment between when the visible action is narrated as well as the time it
occurs in the video.</p>

## Dataset Annotation Process
1. The extraction of actions from the transcripts and their annotation of *visible/ not visible* in the video 
is described in detail in this [other project](https://github.com/OanaIgnat/vlog_action_recognition).
2. The visible actions are temporally annotated using this [open source tool][https://github.com/OanaIgnat/video_annotations] that we built.

## Data format

## Citation


# Run the code

## Setup
* I recommend creating a python virtualenv
* I use Python 3.7.7
* pip install -r requirements.txt

## Data requirements

## Usage
1. Check [`args.py`](args.py) to set the arguments
2. 


