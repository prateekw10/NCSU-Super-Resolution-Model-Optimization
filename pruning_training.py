import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import DatasetFromHdf5
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    FPGMPruner,
    L1FilterPruner,
    L2FilterPruner,
)
# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--sparsity",default=0.5,type = float,help="sparsity of the pruning")
parser.add_argument('--pruner', type=str, default='l1filter',choices=['level', 'l1filter', 'l2filter', 'fpgm'],help='pruner to use')



str2pruner = {
    'level': LevelPruner,
    'l1filter': L1FilterPruner,
    'l2filter': L2FilterPruner,
    'fpgm': FPGMPruner
}
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
    
def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("data/eval.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(size_average=False)
    model.apply(weights_init)
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    print(model)
    model.apply(weights_init)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)
    
    if cuda:
        dummy_input = torch.rand(1, 1, 256, 256).to('cuda')
    else:
        dummy_input = torch.rand(1, 1, 256, 256).to('cpu')
    
    
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(13) ]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, "./onnx/VDSR_prepruning.onnx", verbose=True, input_names=input_names, output_names=output_names)        
    
    # pruning
    pruner_cls = str2pruner[opt.pruner]
    kw_args = {}
    if opt.pruner == 'level':
        config_list = [
            {
                'sparsity': opt.sparsity,
                'op_types': ['default']
            }
        ]

    elif opt.pruner in ('l1filter', 'l2filter', 'fpgm'):
        config_list = [
                {
                    'sparsity': opt.sparsity,
                    'op_types': ['Conv2d'],
                }
            ]

    pruner = pruner_cls(model, config_list, **kw_args)

    # Pruner.compress() returns the masked model
    model = pruner.compress()
    pruner.get_pruned_weights()

    # export the pruned model masks for model speedup
    model_path = 'pruned_model_path/pruned_{}_{}.pth'.format(opt.pruner,opt.sparsity)
    mask_path = 'pruned_mask_path/mask_{}_{}.pth'.format(opt.pruner,opt.sparsity)
    pruner.export_model(model_path=model_path, mask_path=mask_path)
   
    dummy_input = torch.rand(1, 1, 256, 256).to('cpu')
    pruner._unwrap_model()

        
    model.cpu()  
    m_speedup = ModelSpeedup(model, dummy_input, mask_path, 'cpu')
    
    m_speedup.speedup_model()
    print(model)
    print('start finetuning...')    
    
    model.apply(weights_init)
    
    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    print(model)
    
    model.cuda()
    
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint_after_pruniing(model, epoch)


    if cuda:
        dummy_input = torch.rand(1, 1, 256, 256).to('cuda')
    else:
        dummy_input = torch.rand(1, 1, 256, 256).to('cpu')
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(13) ]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, "./onnx/VDSR_postpruning_{}_{}.onnx".format(opt.pruner,opt.sparsity), verbose=True, input_names=input_names, output_names=output_names)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    total_loss = 0

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        # if iteration%100 == 0:
        #     print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))
        total_loss += loss.item()
    print(total_loss/len(training_data_loader))
    
def save_checkpoint_after_pruniing(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}_{}.pth".format(opt.pruner,opt.sparsity)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch.pth"
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    
if __name__ == "__main__":
    main()
