#!/bin/bash

set lower $argv[1]
set upper $argv[2]
set source $argv[3]
set dest $argv[4]

read -P "Enter password: " -s PASSWORD
echo ""

if test -z "$lower"
    echo "argv[1]: lower bound is empty (inclusive)"
    exit 0
end

if test -z "$upper"
    echo "argv[2]: upper bound is empty (inclusive)"
    exit 0
end

if test -z "$source"
    echo "argv[3]: source is empty"
    exit 0
end

if test -z "$dest"
    echo "argv[4]: dest is empty"
    exit 0
end

if [ $upper -lt $lower ]
    echo "the upper bound can't be smaller than the lower bound"
    exit 0
end

for i in (seq $lower $upper)
    # echo "$source/$i"
    # echo "$dest/$i"
    sshpass -p "$PASSWORD" scp -v -r "$source/$i" "$dest/$i" 2>&1 | grep -v debug1
    # scp -r "$source/$i" "$dest/$i"
end

# Example:
# fish manual_scp_transfer.sh 0 99 "/home/user/path/to/rawdata" "remote@111.222.111.222:/D:/path/to/rawdata"
# 
# then input your password to the interactive input 