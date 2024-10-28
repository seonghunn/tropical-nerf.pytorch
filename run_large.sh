#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 {bunny|dragon|happy|armadillo|drill|lucy|all}"
  exit 1
fi

# Run commands with different seeds
case "$1" in
  bunny)
    python -m tropical.stanford.train -e -m large -d $1 -s 31
    ;;
  dragon)
    python -m tropical.stanford.train -e -m large -d $1 -s 13
    ;;
  happy)
    python -m tropical.stanford.train -e -m large -d $1 -s 6
    ;;
  armadillo)
    python -m tropical.stanford.train -e -m large -d $1 -s 2
    ;;
  drill)
    python -m tropical.stanford.train -e -m large -d $1 -s 5
    ;;
  lucy)
    python -m tropical.stanford.train -e -m large -d $1 -s 25
    ;;
  all)
    python -m tropical.stanford.train -e -m large -d bunny -s 31
    python -m tropical.stanford.train -e -m large -d dragon -s 13
    python -m tropical.stanford.train -e -m large -d happy -s 6
    python -m tropical.stanford.train -e -m large -d armadillo -s 2
    python -m tropical.stanford.train -e -m large -d drill -s 5
    python -m tropical.stanford.train -e -m large -d lucy -s 25
    ;;
  *)
    echo "Invalid option."
    exit 1
    ;;
esac

exit 0
