#!/bin/bash
export CYTHON_TRACE=1
pip install -v --editable .[cu12] 2>&1 | tee _build.log