
#通过预测出来的直方图来计算MOS值
import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy import stats
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from EMDLoss import EMDLoss
from utils import score_utils_KON,score_utils_bid ,score_utils
import scipy.stats
from scipy.optimize import curve_fit
import time
from loss import var_loss

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def EMD(y_true, y_pred):
    cdf_ytrue = np.cumsum(y_true, axis=-1)
    cdf_ypred = np.cumsum(y_pred, axis=-1)
    samplewise_emd = np.sqrt(np.mean(np.square(np.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return np.mean(samplewise_emd)

def JSDmetric(y_true, y_pred):
    M=(y_true+y_pred)/2
    js=0.5*scipy.stats.entropy(y_true, M)+0.5*scipy.stats.entropy(y_pred, M)
    return js

def histogram_intersection(h1, h2):
    intersection = 0
    for i in range(len(h1)):
        intersection += min(h1[i], h2[i])
    return intersection
def cosine_similarity(x, y):
    '''
    Cosine Similarity of two tensors
    Args:
        x: torch.Tensor, m x d
        y: torch.Tensor, n x d
    Returns:
        result, m x n
    '''
    assert x.size(1) == y.size(1)
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return x @ y.transpose(0, 1)


class L1RankLoss(torch.nn.Module):
    
    """
    L1 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)
        self.rank_w = kwargs.get("rank_w", 1)
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        return l1_loss

class MemoryGraph(nn.Module):
    def __init__(self,in_features, out_features, num_nodes):
        # B*1024*num_classes
        super(MemoryGraph, self).__init__()
        self.num_nodes = num_nodes

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv1= nn.Conv1d(in_features*2, in_features, 1)

        self.long_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.long_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.fc_eq3_w = nn.Linear(num_nodes, num_nodes)
        self.fc_eq3_u = nn.Linear(num_nodes, num_nodes)
        self.fc_eq4_w = nn.Linear(in_features, in_features)
        self.fc_eq4_u = nn.Linear(in_features, in_features)


    def forward_construct_short(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        # x=x.unsqueeze(-1)
        # x_glb =x
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ###  Conv  to the short memory ###
        x = torch.cat((x_glb, x), dim=1)
        x = self.conv1(x)
        x = torch.sigmoid(x)
        return x

    def forward_construct_long(self,x,short_memory):
        # x=x.unsqueeze(-1)
        with torch.no_grad():
            long_a = self.long_adj(x.transpose(1, 2))
            long_a = long_a.view(-1, x.size(2))
            long_w = self.long_weight(short_memory)
            long_w = long_w.view(x.size(0) * x.size(2), -1)
        x_w = short_memory.view(x.size(0) * x.size(2), -1)  # B*num_c,1024 短期记忆包含全局关系，提取相对权重关系。生成weight
        x_a = x.view(-1, x.size(2))          # B*1024, num_c, 注意力直接，注重提取个体之间出现的关系。生成adj
        # eq(3)
        av = torch.tanh(self.fc_eq3_w(x_a) + self.fc_eq3_u(long_a))
        # eq(4)
        wv = torch.tanh(self.fc_eq4_w(x_w) + self.fc_eq4_u(long_w))
        # eq(5)
        x_a = x_a + av * long_a
        x_a = x_a.view(x.size(0),x.size(2),-1)
        x_w = x_w + wv * long_w
        x_w = x_w.view(x.size(0),x.size(1),x.size(2))
        long_adj = self.long_adj(x_a)
        long_weight = self.long_weight(x_w)
        x = x + short_memory
        long_graph_feature1 = torch.mul(long_adj.transpose(1, 2),x)
        long_graph_feature2 = torch.mul(long_graph_feature1,long_weight)
        long_graph_feature2 = torch.sigmoid(long_graph_feature2)
        return long_graph_feature2

    def forward(self,x):
        short_memory = self.forward_construct_short(x)
        long_memory = self.forward_construct_long(x,short_memory)
        return long_memory


from cvt import get_cls_model

from einops import rearrange
class RichIQA(torch.nn.Module):

    def __init__(self,  options):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.net=get_cls_model().cuda()
        self.AP1=torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.n_features =64+192+384
        self.n_features2=256+self.n_features
        self.layers = nn.Sequential(
            nn.Linear(self.n_features,self.n_features),
            nn.Linear(self.n_features,5),
        )


        self.fc = nn.Conv2d(self.n_features, 5, (1,1), bias=False)
        self.last_linear = nn.Conv1d(256, 5, 1)
        self.conv_transform = nn.Conv2d(self.n_features, 256, (1,1))
        self.mask_mat = nn.Parameter(torch.eye(5).float())
        self.gcn = MemoryGraph(256,256, 5)


    def forward_sam(self, x):
        """
        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)                                           ##(B,C_in,H,W) -> (B,num_c,H,W)
        mask = mask.view(mask.size(0), mask.size(1), -1)            ##(B,num_c,H*W)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)                                 ## mask = (B,H*W,num_c)

        x = self.conv_transform(x)                                  ##  (B,1024,H,W)
        x = x.view(x.size(0), x.size(1), -1)                        ## x = (B,1024,H*W)
        x = torch.matmul(x, mask)                                   ##  (B,1024,num_c)

        return x
    def forward(self, X):#256*64*64
        """Forward pass of the network.
        """
        N = X.size()[0]
        X0 =self.net.stage0(X)
        X0 =X0[0]
        X1 =self.net.stage1(X0)
        X1 =X1[0]
        X2 =self.net.stage2(X1)

        X0=self.AP1(X0).view(N,-1)
        X1=self.AP1(X1).view(N,-1)
        X2=self.AP1(X2[0]).view(N,-1)
        X=torch.cat((X0,X1),dim=1)
        X=torch.cat((X,X2),dim=1)

        v=self.forward_sam(X.unsqueeze(-1).unsqueeze(-1))
        z=self.gcn(v)
        z = z + v
        out2 =torch.nn.functional.normalize(self.AP1(rearrange(z.unsqueeze(-1),'b c h w -> b h c w', h=5, w=1)).view(N,-1))
        
        
        X = torch.nn.functional.normalize(X)
        hist=self.layers(X)
        hist=0.999*hist+0.001*out2
    
        hist=F.softmax(hist,dim=1)



        return hist
class RichIQAManager(object):
    def __init__(self, options, path):

        print('Prepare the network and data.')
        self._options = options
        self._path = path


        self._net = torch.nn.DataParallel(RichIQA(self._options), device_ids=[0]).cuda()
        print(self._net)
        # Criterion.

        self._criterion1 = EMDLoss().cuda()
        self._criterion3 =L1RankLoss().cuda()
        self._var_loss=var_loss(self._options['dataset']).cuda()

        # Solver.
        self._solver = torch.optim.Adam(
                self._net.module.parameters(), lr=self._options['base_lr'],
                weight_decay=self._options['weight_decay'])

        
        
        resize=512
        crop_size = 384
        train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize,resize)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
        ])

            
            
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize,resize)),
            torchvision.transforms.CenterCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

           
        if self._options['dataset'] == 'KONIQ10K':
            import KONIQ10KFolder_sos
            train_data = KONIQ10KFolder_sos.KONIQ10KFolder(
                    root=self._path['KONIQ10K'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = KONIQ10KFolder_sos.KONIQ10KFolder(
                    root=self._path['KONIQ10K'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        import Folder_sos
        if self._options['dataset'] == 'SPAQ':  

            train_data = Folder_sos.SPAQFolder(
                    root=self._path['SPAQ'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = Folder_sos.SPAQFolder(
                    root=self._path['SPAQ'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        elif self._options['dataset'] == 'bid':
            train_data = Folder_sos.BIDFolder(
                    root=self._path['bid'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = Folder_sos.BIDFolder(
                    root=self._path['bid'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        elif self._options['dataset'] == 'livec':
            train_data = Folder_sos.LIVEChallengeFolder(
                    root=self._path['livec'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = Folder_sos.LIVEChallengeFolder(
                    root=self._path['livec'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        elif self._options['dataset'] == 'flive':
            train_data = Folder_sos.FLIVEFolder(
                    root=self._path['flive'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = Folder_sos.FLIVEFolder(
                    root=self._path['flive'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=8, pin_memory=True,drop_last=True)

        

    def train(self):
        """Train the network."""
        print('Training.')
        best_value=0
        best_Cosine = 0.0

        
        best_MOSsrcc = 0.0

        best_epoch_hist = None
        best_epoch_mos = None
        print('Epoch\tTrain loss\tTest_JSD\tTest_EMD\tTest_RMSE\tTest_inter\tTest_Cosine\t\tTest_MOSsrcc\tTest_MOSplcc\tTest_MOSrmse')
        for t in range(self._options['epochs']):
            epoch_loss = []
            import time
            # time.sleep(0.15)
            for X,gtmos, gtsos,y in self._train_loader:
                gtmos =gtmos.to(torch.float32)
                # Data.
                X = X.cuda()
                y = y.cuda()
                gtmos = gtmos.cuda()
                gtsos = gtsos.cuda()
                
                # Clear the existing gradients.
                self._solver.zero_grad()
                prehist=self._net(X)


                loss1 = self._criterion1(prehist, y.view(len(prehist),self._options['numbin']).detach())
                calsos=torch.zeros((prehist.shape[0],1)).cuda()
                calmos=torch.zeros((prehist.shape[0],1)).cuda()
                if self._options['dataset'] == 'KONIQ10K':
                    for j,pre in enumerate(prehist):
                        si = torch.arange(1, 6, 1).cuda()#LIVE

                        calmos[j] = torch.sum(prehist[j] * si)

                        calsos[j] = torch.sqrt( torch.sum(((si -calmos[j]) ** 2) *prehist[j]))
                if self._options['dataset'] == 'bid':
                    for j,pre in enumerate(prehist):
                        si = torch.arange(0.5, 5.5, 1).cuda()#LIVE0.1, 1.10, 0.2
                        calmos[j] = torch.sum(prehist[j] * si)
                        calsos[j] = torch.sqrt( torch.sum(((si -calmos[j]) ** 2) *prehist[j]))
                if (self._options['dataset'] == 'SPAQ')|(self._options['dataset'] == 'livec')|(self._options['dataset'] == 'flive'):
                    for j,pre in enumerate(prehist):
                        si = torch.arange(10, 110, 20).cuda()#LIVE0.1, 1.10, 0.2
                        calmos[j] = torch.sum(prehist[j] * si)
                        calsos[j] = torch.sqrt( torch.sum(((si -calmos[j]) ** 2) *prehist[j]))

                loss3=self._criterion3(calmos, gtmos.detach())#premos gtmos loss
                gtmos=gtmos.unsqueeze(1)
                loss3_sos=self._var_loss(calmos,calsos)

                loss=2*5*10*loss1+5*loss3+0.5*loss3_sos#100:5:0.5=200:10:1
                
                epoch_loss.append(loss.item())
                loss.requires_grad_(True)
                loss.backward()
                self._solver.step()

            JSDtest,EMDtest,RMSEtest,intersectiontest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse = self._consitency(self._test_loader)
            if Cosinetest >= best_Cosine:
                best_Cosine = Cosinetest
                
                best_EMD = EMDtest
                best_RMSE = RMSEtest
                best_inter =intersectiontest
                best_JSD =JSDtest


                best_epoch_hist = t + 1
                print('*', end='')
                pwd = os.getcwd()
            if MOSsrcc >= best_MOSsrcc:
                best_MOSsrcc = MOSsrcc
                    
                best_MOSplcc = MOSplcc
                best_MOSrmse = MOSrmse
                best_epoch_mos = t + 1
                # print('*', end='')
                # pwd = os.getcwd()
                # modelpath = os.path.join('/lustre/home/acct-minxiongkuo/minxiongkuo/gyx/checkpoint/mxk',('net_params_trainkon' +'_train_num='+ str(self._options['train_num'])+'_best' + '.pkl'))
                # torch.save(self._net.state_dict(), modelpath)
                

            print('%d\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t+1, sum(epoch_loss) / len(epoch_loss),  JSDtest,EMDtest,RMSEtest,intersectiontest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse))           

        print('Best at epoch %d: test cosine %f, Best at epoch %d: test srcc %f' % (best_epoch_hist, best_Cosine,best_epoch_mos, best_MOSsrcc))
        
        return best_JSD,best_EMD, best_RMSE,best_inter,best_Cosine, best_MOSsrcc,best_MOSplcc,best_MOSrmse

    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0
        JSD_test = []
        JSD_all=0
        JSDtest=0
        JSD0=0
        
        EMD_test = []
        EMD_all=0
        EMDtest=0
        EMD0=0

        RMSE_all=0
        RMSE0=0
        RMSE_test=[]
        RMSEtest=0

        Cosine_all=0
        Cosine0=0
        Cosine_test=[]
        Cosinetest=0

        intersection_test = []
        intersection_all=0
        intersectiontest=0
        pscores_MOS = []
        tscores_MOS = []
        if self._options['dataset'] == 'bid':
            score_util=score_utils_bid 
        if (self._options['dataset'] == 'SPAQ')|(self._options['dataset'] == 'livec')|(self._options['dataset'] == 'flive'):
            score_util= score_utils
        if self._options['dataset'] == 'KONIQ10K':
            score_util=score_utils_KON
        for X,gtmos,gtsos, y in data_loader:
            # Data.
            X = X.cuda()

            y = y.cuda()
            gtmos= gtmos.cuda()
            # Prediction.
            prehist= self._net(X)
            score=prehist
            score=score[0].cpu()
            y=y[0].cpu()
            gtmos=gtmos[0].cpu()
            pscores_MOS.append(score_util.mean_score(score.detach().numpy()))
            tscores_MOS.append(gtmos.detach().numpy())
            
            ##histogram
            RMSE0=np.sqrt(((score.detach().numpy() - y.detach().numpy()) ** 2).mean())#对于每张直方图，求结果
            EMD0=EMD(score.detach().numpy(),y.detach().numpy())
            JSD0=JSDmetric(score.detach().numpy(),y.detach().numpy())
            intersection0=histogram_intersection(score.detach().numpy(),y.detach().numpy())
            X=[score.detach().numpy(),y.detach().numpy()]
            Cosine0 = (1-pairwise_distances( X, metric='cosine'))[0][1]
            JSD_test.append(JSD0)
            EMD_test.append(EMD0)
            RMSE_test.append(RMSE0)
            intersection_test.append(intersection0)
            Cosine_test.append(Cosine0)
        tscores_MOS=np.squeeze(tscores_MOS)
        pscores_MOS=np.squeeze(pscores_MOS)
        num_total =len(EMD_test)
        for ele in range(0, len(EMD_test)):
            JSD_all = JSD_all + JSD_test[ele]  
            EMD_all = EMD_all + EMD_test[ele]  
            RMSE_all = RMSE_all + RMSE_test[ele]  
            intersection_all = intersection_all + intersection_test[ele] 
            Cosine_all = Cosine_all + Cosine_test[ele] 
        # EMD_all=torch.sum(EMD_test)
        JSDtest=JSD_all/num_total
        EMDtest=EMD_all/num_total
        RMSEtest=RMSE_all/num_total
        intersectiontest=intersection_all/num_total
        Cosinetest=Cosine_all/num_total
        
        ##MOS
        pscores_MOS_logistic = fit_function(tscores_MOS,pscores_MOS)
        MOSsrcc, _ = stats.spearmanr(pscores_MOS_logistic,tscores_MOS)
        MOSplcc, _ = stats.pearsonr(pscores_MOS_logistic,tscores_MOS)
        MOSrmse=np.sqrt((((pscores_MOS_logistic)-np.array(tscores_MOS))**2).mean())
        self._net.train(True)  # Set the model to training phase
        return JSDtest,EMDtest,RMSEtest,intersectiontest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train RichIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32, help='Batch size:8.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training:50.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-7, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='KONIQ10K',
                        help='dataset: KONIQ10K')
    parser.add_argument('--seed',  type=int, default=0)

    args = parser.parse_args()

    seed =25198744
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("seed:", seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset':args.dataset,
        'train_index': [],
        'test_index': [],
        
        'train_num': 0,
        'numbin':5
    }
    
    path = {

        'KONIQ10K': os.path.join('/mnt/sda/gyx/image_database','KON'),
        'bid': os.path.join('/mnt/sda/gyx/image_database','BID'),
        'livec': os.path.join('/mnt/sda/gyx/image_database','CLIVE'),
        'flive': '/mnt/sda/gyx/image_database/FLIVEDatabase/database',
        'SPAQ': '/mnt/sda/gyx/image_database/SPAQ',
    }
    
    
    if options['dataset'] == 'KONIQ10K':          
        index = list(range(0,10073))
        options['numbin'] == 5 
    elif options['dataset'] == 'bid':
        index = list(range(0,586))  
    elif options['dataset'] == 'livec':
        index = list(range(0,1162))
    elif options['dataset'] == 'flive':
        index = list(range(0,39810))
    elif options['dataset'] == 'SPAQ':
        index = list(range(0,11125))
    
    
    EMD_all = np.zeros((1,10),dtype=np.float32)
    RMSE_all = np.zeros((1,10),dtype=np.float32)
    Cosine_all = np.zeros((1,10),dtype=np.float32)
    JSD_all = np.zeros((1,10),dtype=np.float32)
    inter_all = np.zeros((1,10),dtype=np.float32)
    
    MOSsrcc_all = np.zeros((1,10),dtype=np.float32)
    MOSplcc_all = np.zeros((1,10),dtype=np.float32)
    MOSrmse_all = np.zeros((1,10),dtype=np.float32)
    
    for i in range(0,10):
        #randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8*len(index))]
        test_index = index[round(0.8*len(index)):len(index)]
        options['train_num'] = i
        options['train_index'] = train_index
        options['test_index'] = test_index

        manager = RichIQAManager(options, path)
        JSD,EMD, RMSE,inter,Cosine, MOSsrcc,MOSplcc,MOSrmse = manager.train()
        
        EMD_all[0][i] = EMD
        RMSE_all[0][i] = RMSE
        Cosine_all[0][i] = Cosine
        JSD_all[0][i] = JSD
        inter_all[0][i] = inter
        
        #
        MOSsrcc_all[0][i] = MOSsrcc
        MOSplcc_all[0][i] = MOSplcc
        MOSrmse_all[0][i] = MOSrmse

        

        
        
    EMD_mean = np.mean(EMD_all)
    RMSE_mean = np.mean(RMSE_all)
    Cosine_mean = np.mean(Cosine_all)
    JSD_mean = np.mean(JSD_all)
    inter_mean = np.mean(inter_all)
    
    MOSsrcc_mean = np.mean(MOSsrcc_all)
    MOSplcc_mean = np.mean(MOSplcc_all)
    MOSrmse_mean = np.mean(MOSrmse_all)


if __name__ == '__main__':

    main()
