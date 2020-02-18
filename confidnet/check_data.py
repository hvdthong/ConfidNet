from keras.datasets import mnist, cifar10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)
# print(x_train.shape)
print(x_test)

import torch
x_test = torch.load('./data/mnist-data/MNIST/processed/test.pt')
print(x_test)

# from torchvision import datasets
# import torch
# train_dataset = datasets.MNIST(root='./data/mnist-data/', train=True, download=True)
# train_loader = torch.utils.data.DataLoader(
#                 train_dataset,
#                 batch_size=128,
#                 shuffle=True,
#                 pin_memory=False,
#                 num_workers=1,
#             )
# from tqdm import tqdm
# loop = tqdm(train_loader)
# device = torch.device("cuda")
# for batch_id, (data, target) in enumerate(loop):
#     data, target = data.to(device), target.to(device)
#     print(data.shape)
#     print(data)
#     print(target.shape)
#     print(target)
#     exit()
