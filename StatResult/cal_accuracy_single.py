# Cal accuracy for final_faceforensics++ single fake test set


method = 'deepfakes'
# [deepfakes, face2face, faceswap, neuraltextures]
test_1_prediction = open('D:\\软件\\term1-simulator-windows\\FaceForensics\\output_result\\' + method + '_1_prediction.txt',
                         'r')

result_1_1 = 0
result_1_0 = 0

for line in test_1_prediction:
    print(str(line)[21])
    if str(line)[21] == '1':
        result_1_1 += 1
    else:
        result_1_0 += 1

print('result_1_1: %d' % result_1_1)
print('result_1_0: %d' % result_1_0)
