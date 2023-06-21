#!/bin/sh

git add .
if [ $# -eq 0 ]; then
    git commit -m "Update"
else
    git commit -m "$1"
fi
git push