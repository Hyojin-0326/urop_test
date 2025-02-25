#!/bin/bash

git status
git add .

read -p "Enter commit message: " commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Auto commit: $(date +"%Y-%m-%d %H:%M:%S")"
fi

git commit -m "$commit_msg"
git push origin main

echo "Git 커밋 & 푸시 완료! (메시지: '$commit_msg')"
