import argparse
import matplotlib

matplotlib.use('Agg')

from model_utils import *
from utilities import *
from preprocessIMDB import process_test_data, process_train_data
from preprocessSNLI import process_snli_train_data, process_snli_test_data

parser = argparse.ArgumentParser(description='Eqprop')
parser.add_argument('--dataset', type=str, default='IMDB', metavar='t', help='dataset')

parser.add_argument('--fc_layers', nargs='+', type=int, default=[1000, 40, 2], metavar='A',
                    help='Number of fully connected layers in the network')
parser.add_argument('--seqLen', nargs='+', type=int, default=[600], metavar='A',
                    help='Sequence Length of the data')

parser.add_argument('--act', type=str, default='mysig', metavar='a', help='activation function')
parser.add_argument('--lrs', nargs='+', type=float, default=[], metavar='l', help='layer wise lr (add the lr of projection layers at the beginning then the fc layers')
parser.add_argument('--alg', type=str, default='EP', metavar='al', help='EP or BPTT')
parser.add_argument('--batch_size', type=int, default=20, metavar='M', help='Batch Size')
parser.add_argument('--T', type=int, default=20, metavar='T', help='Time of first phase')
parser.add_argument('--K', type=int, default=4, metavar='K', help='Time of second phase')
parser.add_argument('--thirdphase', default=False, action='store_true', help='for better gradient estimates')
parser.add_argument('--betas', nargs='+', type=float, default=[0.0, 0.01], metavar='Bs', help='Betas')
parser.add_argument('--epochs', type=int, default=1, metavar='EPT', help='Number of epochs per tasks')
parser.add_argument('--check-thm', default=False, action='store_true', help='checking the gdu while training')
parser.add_argument('--lr-decay', default=False, action='store_true', help='enabling learning rate decay')
parser.add_argument('--save', default=False, action='store_true', help='saving results')
parser.add_argument('--execute', type=str, default='train', metavar='tr', help='training or plot gdu curves')
parser.add_argument('--load-path', type=str, default='', metavar='l', help='load a saved model')
parser.add_argument('--seed', type=int, default=None, metavar='s', help='random seed')
parser.add_argument('--device', type=int, default=0, metavar='d', help='device')

args = parser.parse_args()
command_line = ' '.join(sys.argv)
if args.seed is not None:
    torch.manual_seed(args.seed)
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

if args.save:
    if args.load_path == '':
        path = 'results/' + args.dataset + '/' + datetime.now().strftime('%Y-%m-%d') + '/' + datetime.now().strftime('%H-%M-%S') + '_gpu' + str(args.device)
    else:
        path = args.load_path
    if not (os.path.exists(path)):
        os.makedirs(path)
else:
    path = ''

print('Default dtype :\t', torch.get_default_dtype(), '\n')

if args.dataset == 'IMDB':
    embed_dim = 300
    args.fc_layers = [args.seqLen[0] * embed_dim] + args.fc_layers
    if args.execute == 'train':
        train_loader = process_train_data(args.seqLen[0], batch_size=args.batch_size)
    test_loader = process_test_data(args.seqLen[0], batch_size=args.batch_size)

if args.dataset == 'SNLI':
    embed_dim = 300
    args.fc_layers = [4 * args.seqLen[0] * embed_dim] + args.fc_layers  #There are 4 projection layers for this model as described in the paper
    if args.execute == 'train':
        train_loader = process_snli_train_data(args.seqLen[0], batch_size=args.batch_size)
    test_loader = process_snli_test_data(args.seqLen[0], batch_size=args.batch_size)

if args.act == 'my_sigmoid':
    activation = my_sigmoid
elif args.act == 'sigmoid':
    activation = torch.sigmoid
elif args.act == 'tanh':
    activation = torch.tanh
elif args.act == 'hard_sigmoid':
    activation = hard_sigmoid
elif args.act == 'my_hard_sig':
    activation = my_hard_sig

criterion = torch.nn.MSELoss(reduction='none').to(device)       #loss Function used MSE LOss

if args.load_path == '':
    if args.dataset == 'IMDB':
        model = IMDB_model(args.fc_layers, activation=activation, seqLen=args.seqLen[0])
    if args.dataset == 'SNLI':
        model = P_MLP(args.fc_layers, activation=activation, seqLen=args.seqLen[0])
else:
    model = torch.load(args.load_path + '/model.pt', map_location=device)

model.to(device)
print(model)

betas = args.betas[0], args.betas[1]

if args.execute == 'train':
    assert (len(args.lrs) == len(model.synapses))

    # Constructing the optimizer
    optim_params = []

    for idx in range(len(model.synapses)):
        optim_params.append({'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx]})

    optimizer = torch.optim.Adam(optim_params, eps=1e-8)
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)
    else:
        scheduler = None

    checkpoint = None
    print(optimizer)
    print('\ntraining algorithm : ', args.alg, '\n')

    train(model, optimizer, train_loader, test_loader, args.T, args.K, betas, device, args.epochs, criterion,
          alg=args.alg,
          check_thm=args.check_thm, save=args.save, path=path, checkpoint=checkpoint,
          thirdphase=args.thirdphase, scheduler=scheduler)


elif args.execute == 'gducheck':
    print('\ntraining algorithm : ', args.alg, '\n')
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images, labels = images[0:20, :], labels[0:20]
    images, labels = images.to(device), labels.to(device)
    BPTT, EP = check_gdu(model, images, labels, args.T, args.K, betas, criterion, alg=args.alg)
    if args.thirdphase:
        beta_1, beta_2 = args.betas
        _, EP_2 = check_gdu(model, images, labels, args.T, args.K, (beta_1, -beta_2), criterion, alg=args.alg)

    RMSE(BPTT, EP)
    if args.save:
        bptt_est = get_estimate(BPTT)
        ep_est = get_estimate(EP)
        if args.thirdphase:
            ep_2_est = get_estimate(EP_2)
            compare_estimate(bptt_est, ep_est, ep_2_est, path)
            plot_gdu(BPTT, EP, path, EP_2=EP_2, alg=args.alg)
        else:
            plot_gdu(BPTT, EP, path, alg=args.alg)

elif args.execute == 'test':
    print('---Testing model---')
    test_acc = test_model(model, test_loader, args.T, device)
    test_acc /= len(test_loader.dataset)
    print('\nTest accuracy :', round(test_acc, 2))