# Simple Baseline
The goal of our simple baseline is to make sure that the evaluation scripts is working. Thus we implemented a simple model to randomly generate sql queries given table schema, and to create further baseline on AGG predictor using keyword matching.

## Base-line results
Dev accuracy_QueryMatch: 0.003
  breakdown on (agg, sel, where): [0.174 0.123 0.025]
Dev execution accuracy: 0.009
Test accuracy_QueryMatch: 0.001
  breakdown on (agg, sel, where): [0.149 0.122 0.016]
Test execution accuracy: 0.015

## Further baseline results
Train accuracy_QueryMatch: 0.823
Dev accuracy_QueryMatch: 0.795

This is an expected result because it's randomly selecting aggregators and rows. We can also find out that:
1. The dataset is evenly distributed in terms of different aggregators.
2. The are 8 columns per table on averagge.
3. On dev set, 1/4 of the queries doesn't have condition, whereas it's 1/6 on test set.

# Evaluation Method
There are two ways to evaluate the performance:

### 1. Execution Accuracy: 
Execute the generated query and validate the query result. The higher, the better
### 2. Query Matching Accuracy:
2. Compare the generated query stringg with ground truth query. The higher, the better

## How to run
1. unzip data.zip:
2. execute simple_baseline.py to generate the simple model:
python2.7 simple_baseline.py
3. run score.py and specify the model path
python2.7 score.py --model simple_model.pickle

## Prerequisites:
1. The program runs on python2.7
2. Make sure data/ folder is located in the same directory as score.py and simple_baseline.py
3. Make sure to specify correct path of simple_model(dumped by pickle)
4. records package version should be 0.5.1(latest version would cause error)

After run, it will display the two kinds of accuracy on dev and test sets repectively, as well as the further baseline results of AGG predictor.





