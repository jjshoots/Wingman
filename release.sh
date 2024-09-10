#!/bin/bash
python3 -m build . && twine upload dist/*
rm -rf dist/
