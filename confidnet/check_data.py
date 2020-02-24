from keras.datasets import mnist, cifar10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# # print(x_train)
# # print(x_train.shape)
# print(x_test)

# import torch
# x_test = torch.load('./data/mnist-data/MNIST/processed/test.pt')
# print(x_test)

from torch.utils import data
import torch
print(type(x_train))
tensor_x = torch.Tensor(x_train) # transform to torch tensor
tensor_y = torch.Tensor(y_train)

my_dataset = data.TensorDataset(tensor_x) # create your datset
my_dataloader = data.DataLoader(my_dataset, batch_size=128, shuffle=True) # create your dataloader

from tqdm import tqdm
loop = tqdm(my_dataloader)
device = torch.device("cuda")
for batch_id, batch_data in enumerate(loop):
    
    print(batch_data[0].shape)

    # print(target.shape)
    exit()

# from torchvision import datasets
# from torch.utils import data
# import torch
# # train_dataset = datasets.MNIST(root='./data/mnist-data/', train=True, download=True)
# # print(type(train_dataset))
# # train_dataset = data.TensorDataset(train_dataset)
# train_dataset = torch.load('./data/mnist-data/MNIST/processed/test.pt')
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
# for batch_id, (x, y) in enumerate(loop):
#     x, y = x.to(device), y.to(device)
#     print(x)
#     print(y)
#     print(x.shape)
#     print(y.shape)
#     exit()
