#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 {bunny|dragon|happy|armadillo|drill|lucy|all}"
  exit 1
fi

# Run commands with different seeds
case "$1" in
  bunny)
    python -m tropical.stanford.train -e -m small -d $1 -s 1
    ;;
  dragon)
    python -m tropical.stanford.train -e -m small -d $1 -s 4
    ;;
  happy)
    python -m tropical.stanford.train -e -m small -d $1 -s 2
    ;;
  armadillo)
    python -m tropical.stanford.train -e -m small -d $1 -s 1
    ;;
  drill)
    python -m tropical.stanford.train -e -m small -d $1 -s 9
    ;;
  lucy)
    python -m tropical.stanford.train -e -m small -d $1 -s 13
    ;;
  all)
    python -m tropical.stanford.train -e -m small -d bunny -s 1
    python -m tropical.stanford.train -e -m small -d dragon -s 4
    python -m tropical.stanford.train -e -m small -d happy -s 2
    python -m tropical.stanford.train -e -m small -d armadillo -s 1
    python -m tropical.stanford.train -e -m small -d drill -s 9
    python -m tropical.stanford.train -e -m small -d lucy -s 13
    ;;
  *)
    echo "Invalid option."
    exit 1
    ;;
esac

exit 0
