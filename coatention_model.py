import torch
import torch.nn as nn
import ConvGRU
import torch.nn.functional as F
from unet import UNet
from PAM import PAM_Module


relu = torch.nn.ReLU()


class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())

    def forward(self, X):
        return self.step(X)


class CoattentionModel(nn.Module):
    def __init__(self, num_classes=1, all_channel=256, all_dim=60):  # 473./8=60
        super(CoattentionModel, self).__init__()
        self.encoder = UNet()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.sa = PAM_Module(all_channel)
        self.dim = all_dim * all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, all_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.conv_fusion = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=True)
        self.relu_fusion = nn.ReLU(inplace=True)
        self.prelu = nn.ReLU(inplace=True)
        self.relu_m = nn.ReLU(inplace=True)
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

        # self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        # self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()
        self.propagate_layers = 3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2, input3):  # Note input2 can be multi-frame image

        # input1_att, input2_att = self.coattention(input1, input2)

        input_size = input1.size()[2:]
        batch_num = input1.size()[0]
        # node0=256*64*64   x2_0=128*128*128   x1_0=64*256*256

        nodes0, x2_0, x1_0 = self.encoder(input1)
        nodes1, x2_1, x1_1 = self.encoder(input2)
        nodes2, x2_2, x1_2 = self.encoder(input3)

        nodes0 = self.conv51(self.sa(nodes0))
        nodes1 = self.conv51(self.sa(nodes1))
        nodes2 = self.conv51(self.sa(nodes2))

        x1s = torch.zeros(batch_num, 1, input_size[0], input_size[1]).cuda()
        x2s = torch.zeros(batch_num, 1, input_size[0], input_size[1]).cuda()
        x3s = torch.zeros(batch_num, 1, input_size[0], input_size[1]).cuda()
        # start_time = time.time()
        for ii in range(batch_num):
            node0 = nodes0[ii, :, :, :][None].contiguous().clone()
            node1 = nodes1[ii, :, :, :][None].contiguous().clone()
            node2 = nodes2[ii, :, :, :][None].contiguous().clone()
            # print('size:', query.size())
            for passing_round in range(self.propagate_layers):

                att1_1 = self.generate_attention(node0, node1)
                att1_2 = self.generate_attention(node0, node2)
                att1_cat = torch.cat([att1_1, att1_2], 1)
                attention1 = self.conv_fusion(att1_cat)  # message passing with concat operation

                attention2 = self.conv_fusion(torch.cat([self.generate_attention(node1, node0),
                                                         self.generate_attention(node1, node2)], 1))
                attention3 = self.conv_fusion(torch.cat([self.generate_attention(node2, node0),
                                                         self.generate_attention(node2, node1)], 1))

                h_v1 = self.ConvGRU(attention1, node0)
                # h_v1 = self.relu_m(h_v1)
                h_v2 = self.ConvGRU(attention2, node1)
                # h_v2 = self.relu_m(h_v2)
                h_v3 = self.ConvGRU(attention3, node2)
                # h_v3 = self.relu_m(h_v3)
                node0 = h_v1.clone()
                node1 = h_v2.clone()
                node2 = h_v3.clone()

                # print('attention size:', attention3[None].contiguous().size(), exemplar.size())
                if passing_round == self.propagate_layers - 1:
                    x1s[ii, :, :, :] = self.my_fcn(h_v1, nodes0[ii, :, :, :][None].contiguous(), input_size,
                                                   x2_0[ii].unsqueeze(0),
                                                   x1_0[ii].unsqueeze(0))
                    x2s[ii, :, :, :] = self.my_fcn(h_v2, nodes1[ii, :, :, :][None].contiguous(), input_size,
                                                   x2_1[ii].unsqueeze(0),
                                                   x1_1[ii].unsqueeze(0))
                    x3s[ii, :, :, :] = self.my_fcn(h_v3, nodes2[ii, :, :, :][None].contiguous(), input_size,
                                                   x2_2[ii].unsqueeze(0),
                                                   x1_2[ii].unsqueeze(0))

        # end_time = time.time()
        # print('network fedforward time:', end_time-start_time)
        return x1s, x2s, x3s

    def message_fun(self, input):
        input1 = self.conv_fusion(input)
        input1 = self.relu_fusion(input1)
        return input1

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]
        channel = query.size()[2:]
        #		 #all_dim = exemplar.shape[1]*exemplar.shape[2]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0] * fea_size[1])  # N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0] * fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        # query_att = torch.bmm(exemplar_flat, A).contiguous() #Note that we have this place to interact with the residual structure
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        # input2_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        # input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        # input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        # input2_att = input2_att * input2_mask

        return input1_att

        # print('h_v size, h_v_org size:', torch.min(input1_att), torch.min(exemplar))

    def my_fcn(self, input1_att, exemplar, input_size, x2, x1):  # exemplar,

        input1_att = torch.cat([input1_att, exemplar], 1)
        input1_att = self.conv1(input1_att)
        input1_att = self.bn1(input1_att)
        input1_att = self.prelu(input1_att)

        xu3 = self.upconv3(input1_att)
        xu33 = torch.cat([xu3, x2], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, x1], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        # x1 = self.main_classifier1(input1_att)
        # x1 = F.upsample(x1, input_size, mode='bilinear')  # upsample to the size of input image, scale=8

        x7 = self.softmax(out)

        return x7  # , x2, temp  #shape: NxCx


def Res_Deeplab(num_classes=2):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes - 1)
    return model


def GNNNet(num_classes=2):
    model = CoattentionModel(Bottleneck, [3, 4, 23, 3], num_classes - 1)

    return model
