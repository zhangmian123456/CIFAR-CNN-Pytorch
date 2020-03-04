"""2"""
import random

splitPath = 'split_text.txt'
ttLabels = []
TEST_RATE = 0.2
DATASET_SIZE = 50000

iList = []
ttList = []

for i in range(DATASET_SIZE):
    iList.append(i)
    ttList.append(1)

randomChoose = random.sample(iList, int(DATASET_SIZE * TEST_RATE))

for testNum in randomChoose:
    ttList[testNum] = 0

for i in range(DATASET_SIZE):
    ttLabels.append(str(i) + ' ' + str(ttList[i]) + '\n')

with open(splitPath, 'w') as f:
    f.writelines(ttLabels)
