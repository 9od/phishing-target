import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

import os

from dataset_res import Config, SiameseNetworkDataset, TestYOLO_Dataset
from model_res import SiameseNetwork, ContrastiveLoss
from visualization_res import imshow, show_plot


# ============================Train=========================================
def Train():
    random.seed(10)
    torch.manual_seed(10)

    # NetSet
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=Config.learning_rate)

    # TrainSet
    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.RandomApply([
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.RandomResizedCrop((100, 100)),
                                                transforms.RandomRotation(180),
                                            ], p=0.0),
                                            should_invert=False)
    train_dataloader = DataLoader(siamese_dataset, num_workers=0,
                                  batch_size=Config.train_batch_size, shuffle=True)

    # TestSet
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            should_invert=False)
    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)

    counter = []
    loss_history = []
    loss_test = []
    iteration_number = 0

    dict_img = {}
    dict_freq = {}
    dict_pair = {}

    for epoch in range(0, Config.train_number_epochs):
        count = 0
        for i, data in enumerate(train_dataloader, 0):
            count = count + 1
            img_str0, img_str1, img0, img1, label = data

            update_dict_pair(dict_pair, img_str0, img_str1, img0, img1)
            update_dict_img(dict_img, img_str0, img0)
            update_dict_img(dict_img, img_str1, img1)

            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            update_dict_freq(dict_freq, img_str0)
            update_dict_freq(dict_freq, img_str1)

            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i == len(train_dataloader) - 1:
                # train loss
                print("Epoch number {}\n Train loss {}".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

                # test loss
                loss_t = 0
                for i, data in enumerate(test_dataloader, 0):
                    img_str0, img_str1, x0, x1, label = data
                    x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
                    output1, output2 = net(x0, x1)
                    loss_contrastive = criterion(output1, output2, label)
                    loss_t += loss_contrastive.item()
                print(" Test loss {}\n".format(loss_t / len(test_dataloader)))
                loss_test.append(loss_t / len(test_dataloader))

        print("iteration number", count)

    show_plot(counter, loss_history, loss_test)
    return net, dict_img, dict_freq, dict_pair


def update_dict_pair(dict_pair, img_str0, img_str1, img0, img1):
    for i in range(len(img_str0)):
        key = img_str0[i] + '&' + img_str1[i]
        value = (img0[i], img1[i])
        dict_pair[key] = value


def update_dict_freq(dict_freq, img_str):
    for i in range(len(img_str)):
        key = img_str[i]
        if key in dict_freq:
            dict_freq[key] = dict_freq[key] + 1
        else:
            dict_freq[key] = 1


def update_dict_img(dict_img, img_str, img):
    for i in range(len(img_str)):
        key = img_str[i]
        value = img[i]
        dict_img[key] = value

# =================================Test=======================================
def Test(net):
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            should_invert=False)
    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(10):
        _, x1, _ = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))


# ==============================ScatterVisualization=====================================
def ScatterVisualization(net, dict_img, dict_pair, dataset, range_, ):
    if dataset == "train":
        # folder_dataset_test = dset.ImageFolder(root=Config.training_dir)
        figname = "fig/train.png"
    elif dataset == "test":
        # folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
        figname = "fig/test.png"
    # siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
    #                                         should_invert=False)
    # test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)

    plt.figure()
    # for k in range(0, range_):

    count = 0
    for index, key in enumerate(dict_pair):
        count = count + 1
        if count > 200:
            break

        value = dict_pair[key]
        img0_path = key[:key.find("&")]
        tmp_x0 = value[0]
        img1_path = key[key.find("&")+1:]
        tmp_x1 = value[1]

        x0 = torch.rand(1, 3, 100, 100)
        x0[0] = tmp_x0
        x1 = torch.rand(1, 3, 100, 100)
        x1[0] = tmp_x1

        # img_str0, img_str1, x0, x1, label = data
        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)

        x = 3 * np.random.rand(1)
        y = euclidean_distance.item()

        img0 = transforms.ToPILImage()(x0.squeeze(0))
        img1 = transforms.ToPILImage()(x1.squeeze(0))

        is_same_category = check_label(img0_path, img1_path)
        if is_same_category == 1:
            plt.scatter(x, y, marker='.', color='c', s=10)
            if euclidean_distance < 2:
                save_image_pair("same_euclidean_true/", img0, img1, index, y)
            else:
                save_image_pair("same_euclidean_false/", img0, img1, index, y)
        elif is_same_category == 0:
            plt.scatter(x, y, marker='.', color='r', s=10)
            if euclidean_distance < 2:
                save_image_pair("diff_euclidean_false/", img0, img1, index, y)
            else:
                save_image_pair("diff_euclidean_true/", img0, img1, index, y)

    plt.savefig(figname)
    plt.show()


def save_image_pair(folder_name, img0, img1, index, y):
    pathname = folder_name + str(y) + "+" + str(index + 1) + "/"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    img0.save(pathname + "1.png")
    img1.save(pathname + "2.png")


def check_label(img0_path, img1_path):
    category0 = extract_category(img0_path)
    category1 = extract_category(img1_path)
    return category0 == category1


def extract_category(img_path):
    index0 = img_path.find("mix_train") + 10
    index1 = img_path.find("\\")

    category = img_path[index0: index1]

    return category

# ===============================ApplyPhish=========================================
def ApplyPhish(net):
    folder_dataset_phish = dset.ImageFolder(root=Config.phish_dir)
    phish_dataset = TestYOLO_Dataset(imageFolderDataset=folder_dataset_phish,
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     should_invert=False)
    phish_dataloader = DataLoader(phish_dataset, num_workers=0, batch_size=1, shuffle=False)
    phish_dataiter = iter(phish_dataloader)
    x0, _ = next(phish_dataiter)
    folder_dataset_target = dset.ImageFolder(root=Config.targetlist_dir)
    target_dataset = TestYOLO_Dataset(imageFolderDataset=folder_dataset_target,
                                      transform=transforms.Compose([transforms.ToTensor()]),
                                      should_invert=False)
    target_dataloader = DataLoader(target_dataset, num_workers=0, batch_size=1, shuffle=False)
    dataiter = iter(target_dataloader)
    dissimilarity = []
    brand = []
    for i in range(len(target_dataset)):
        x1, x1name = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
        dissimilarity.append(euclidean_distance.item())
        brand.append(x1name)
    inde = dissimilarity.index(min(dissimilarity))
    print(brand[inde][0])


if __name__ == '__main__':
    net, dict_img, dict_freq, dict_pair = Train()
    ScatterVisualization(net, dict_img, dict_pair, "train", 3)
    # ScatterVisualization(net, "test", 30)
