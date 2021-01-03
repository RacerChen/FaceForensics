# Cal accuracy for final_faceforensics++ test set

test_0_prediction = open('D:\\软件\\term1-simulator-windows\\FaceForensics\\output_result\\test_0_prediction.txt', 'r')
test_1_prediction = open('D:\\软件\\term1-simulator-windows\\FaceForensics\\output_result\\test_1_prediction.txt', 'r')

result_0_0 = 0
result_0_1 = 0
result_1_1 = 0
result_1_0 = 0

for line in test_0_prediction:
    print(str(line)[21])
    if str(line)[21] == '0':
        result_0_0 += 1
    else:
        result_0_1 += 1

for line in test_1_prediction:
    print(str(line)[21])
    if str(line)[21] == '1':
        result_1_1 += 1
    else:
        result_1_0 += 1

print('result_0_0: %d' % result_0_0)
print('result_0_1: %d' % result_0_1)
print('result_1_1: %d' % result_1_1)
print('result_1_0: %d' % result_1_0)
