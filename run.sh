#!/bin/bash

network=wikipedia
clf_mode=dynamic
classifier=MLP

if [ "$clf_mode" = "static" ]
then
  python src/static_n_clf.py --network "$network" --clf "$classifier"
elif [ "$clf_mode" = "dynamic" ]
then
  python src/dynamic_n_clf.py --network "$network" --clf "$classifier"
else
  echo "Undefined setting option!"
fi



