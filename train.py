import numpy as np
import torch
import os
from coatention_model import CoattentionModel
from unet import UNet
from torch.autograd import Variable
import imgaug.augmenters as iaa
from pathlib import Path
from train_dataset import LungDataset
from tqdm import tqdm

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


seq = iaa.Sequential([
    iaa.Affine(rotate=[90, 180, 270]),  # rotate up to 45 degrees
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
])

train_path = Path("/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train")
train_dataset = LungDataset(train_path, seq)

# target_list = []
# for _, label in tqdm(train_dataset):
#     # Check if mask contains a tumorous pixel:
#     if np.any(label):
#         target_list.append(1)
#         label_squeeze=label[0,:,:]
#     else:
#         target_list.append(0)
# np.save('./target_list.npy',target_list)

target_list = np.load('/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/target_list.npy')
uniques = np.unique(target_list, return_counts=True)
fraction = uniques[1][0] / uniques[1][1]

weight_list = []

for target in tqdm(target_list):
    if target == 0:
        weight_list.append(1)
    else:
        weight_list.append(fraction)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))
batch_size = 4  # TODO
num_workers = 8  # TODO
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

model = CoattentionModel().cuda()
unet_pretrained = torch.load(
    "/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/ct_suv/logs/epoch_40.pth",
    map_location='cuda')
weights = unet_pretrained['model']
new_params = model.state_dict().copy()
for i in new_params:
    i_parts = i.split('.')
    if i_parts[0] == 'encoder':
        new_params[i] = weights[".".join(i_parts[1:])]
    elif i_parts[0] in ['upconv3', 'd31', 'd32', 'upconv4', 'd41', 'd42', 'outconv']:
        new_params[i] = weights[".".join(i_parts[:])]

model.load_state_dict(new_params)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
logger = open('./logs/logs.txt', 'a')
resume = 17
if resume:
    state_model = torch.load('/home/fumcomp/Desktop/poorsoltani/min_distance_neighbour/epoch_' + str(resume - 1) + '.pth')
    model.load_state_dict(state_model['model'])
    optimizer.load_state_dict(state_model['optimizer'])
EPOCHE = 30
loss_report = 0
for i in range(resume, EPOCHE):
    print(resume)
    batch_losses = []
    step_100_loss = []
    cnt = 1
    # progress_bar=tqdm(enumerate(train_loader), total=(len(train_loader.batch_sampler)))
    for step, (img, label) in tqdm(enumerate(train_loader), total=(len(train_loader.batch_sampler))):
        node0 = Variable(img[0].requires_grad_()).cuda()
        node1 = Variable(img[1].requires_grad_()).cuda()
        node2 = Variable(img[2].requires_grad_()).cuda()

        label0 = Variable(label[0].requires_grad_()).cuda()
        label1 = Variable(label[1].requires_grad_()).cuda()
        label2 = Variable(label[2].requires_grad_()).cuda()

        pred0, pred1, pred2 = model(node0, node1, node2)
        loss0 = loss_fn(pred0, label0)
        loss1 = loss_fn(pred1, label1)
        loss2 = loss_fn(pred2, label2)
        loss = (loss0 + loss1 + loss2)

        optimizer.zero_grad()  # (reset gradients)
        loss.backward()  # (compute gradients)
        optimizer.step()

        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        step_100_loss.append(loss_value)
        if not (cnt % 100):
            logger.write('step= {}\t mean loss={}\n'.format(step, np.mean(batch_losses)))
            logger.flush()

        cnt += 1

    logger.write('###########################################################\n')
    logger.write('epoch= {}\t mean loss={}\n'.format(i, np.mean(batch_losses)))
    logger.write('###########################################################\n')
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, '/home/fumcomp/Desktop/poorsoltani/min_distance_neighbour/epoch_' + str(i) + '.pth')
    print('##########################################')
    print('################### epoch {} ############'.format(i))
    print('##########################################')
