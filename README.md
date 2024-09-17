# RAG News Retrieval System

![](https://github.com/danzhechen/ragnews/workflows/tests/badge.svg)

## Overview

ragnews is a Python-based question-answering system built using the Groq API. It enhances the capabilities of Groq models by employing Retrieval-Augmented Generation (RAG). The system retrieves and processes news articles from a user-provided database, utilizing them to answer queries with a focus on delivering accurate and up-to-date responses to time-sensitive questions.

## Prerequisites

Before running the project, make sure you have the following:

- Python 3.9
- SQLite (already included with Python)
- Internet access (to download articles and communicate with the LLM API)

## Setup Instructions
1. Create and activate a virtual environment:

```
$ python3.9 -m venv venv
$ . ./venv/bin/activate
```

2. Install Packages:
```
$ pip install -r requirements.txt
$ pip install ./metahtml
```

3. Configure environment variables:
Create a .env file and add your GROQ_API_KEY
```
GROQ_API_KEY=your_api_key_here
$ export $(cat .env)
```

## Import your question

```
$ python3 ragnews.py
ragnews> Who is the current democratic presidential nominee
Based on the provided articles, Kamala Harris is the Vice President and the current Democratic Presidential nominee.
```
