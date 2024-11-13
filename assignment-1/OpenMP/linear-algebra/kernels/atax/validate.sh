#!/bin/bash

./dump.sh "$1"
diff -q ./dump/mini/"$1" ./dump/mini/SEQUENTIAL
diff -q ./dump/small/"$1" ./dump/small/SEQUENTIAL
diff -q ./dump/standard/"$1" ./dump/standard/SEQUENTIAL
diff -q ./dump/large/"$1" ./dump/large/SEQUENTIAL
