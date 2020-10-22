import re

def tokenize_numbers(text):
    return re.sub(r"\d+", " SOMEBUNBER ", text)
