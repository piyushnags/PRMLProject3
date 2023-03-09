'''
Start code for Project 3
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Piyush Nagasubramaniam
    PSU Email ID: pvn5119@psu.edu
    Description: (A short description of what each of the functions you've written does.).
}
'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from util import *
from models import *

def test(model, test_loader, device, criterion):
    """
    Test the model.
    Args:
        model (torch.nn.Module): Model to test.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to use (cuda or cpu).
        criterion (torch.nn.Module): Loss function to use.
    Returns:
        test_loss (float): Average loss on the test set.
        test_acc (float): Average accuracy on the test set.
        preds (numpy array): Class predictions.
        targets (numpy array): Target values.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        preds = []
        targets = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return test_loss, test_acc, preds, targets

#  Note log_interval doesn't actually log to a file but is used for printing. This can be changed if you want to log to a file.
def train(model, train_loader, optimizer, criterion, epochs, 
          log_interval, device, log_dir=None):
    """
    Train the model and periodically log the loss and accuracy.
    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        criterion (torch.nn.Module): Loss function to use.
        epochs (int): Number of epochs to train for.
        log_interval (int): Print loss every log_interval epochs.
        device (torch.device): Device to use (cuda or cpu).
    Returns:
        per_epoch_loss (list): List of loss values per epoch.
        per_epoch_acc (list): List of accuracy values per epoch.
    """
    model.train()
    per_epoch_loss = []
    per_epoch_acc = []
    for epoch in range(epochs):
        train_loss = 0
        preds = []
        targets = []
        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Get the accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Save the predictions and targets if it's the last epoch
            if epoch == epochs - 1:
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        per_epoch_acc.append(train_acc)
        if (epoch+1) % log_interval == 0:
            print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch+1, train_loss, train_acc))
            
            # Save Checkpoints every log_interval epochs
            if log_dir is not None:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                ckpt_path = os.path.join(log_dir, 'ckpt_{}.ckpt'.format(epoch+1))
                torch.save({
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "epoch":epoch,
                    "per_epoch_loss":per_epoch_loss,
                    "per_epoch_acc":per_epoch_acc,
                }, ckpt_path)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return model, np.array(per_epoch_loss), np.array(per_epoch_acc), preds, targets   

def wallpaper_main(args):
    """
    Main function for training and testing the wallpaper classifier.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_classes = 17
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load the wallpaper dataset
    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: Augment the training data given the transforms in the assignment description.
    preprocess = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]
    if args.aug_train:
        augmentation = [
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomCrop(size=(args.img_size, args.img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(1,2)),
        ]
        augmentation = preprocess + augmentation


    # Compose the transforms that will be applied to the images. Feel free to adjust this.
    transform = transforms.Compose(preprocess)
    train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    if args.aug_train:
        augment = transforms.Compose(augmentation)
        aug_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=augment)
        datasets = [train_dataset, aug_dataset]
        train_dataset = ConcatDataset(datasets)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images.")
    # Initialize the model, optimizer, and loss function
    model = CNN2(input_channels=1, img_size=args.img_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train + test the model
    model, per_epoch_loss, per_epoch_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, args.num_epochs, 
                                                                             args.log_interval, device, args.log_dir )
    test_loss, test_acc, test_preds, test_targets = test(model, test_loader, device, criterion)

    # Get stats 
    classes_train, overall_train_mat = get_stats(train_preds, train_targets, num_classes)
    classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)


    print(f'\n\nTrain accuracy: {per_epoch_acc[-1]*100:.3f}')
    print(f'Test accuracy: {test_acc*100:.3f}')

    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats')):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats'))
    overall_file_name = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats', 'overall.npz')

    np.savez(overall_file_name, classes_train=classes_train, overall_train_mat=overall_train_mat, 
                classes_test=classes_test, overall_test_mat=overall_test_mat, 
                per_epoch_loss=per_epoch_loss, per_epoch_acc=per_epoch_acc, 
                test_loss=test_loss, test_acc=test_acc)

    # Note: The code does not save the model but you may do so if you choose with the args.save_model flag.
    if args.save_model:
        model_dir = os.path.join( args.save_dir, 'cnn.pth' )
        torch.save(model.state_dict(), model_dir) 

def taiji_main(args):
    """
    Main function for training and testing the taiji classifier.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_subs = args.num_subs
    num_forms = 46 # Number of taiji forms, hardcoded :p
    sub_train_acc = np.zeros(num_subs)
    sub_class_train = np.zeros((num_subs, num_forms))
    sub_test_acc = np.zeros(num_subs)
    sub_class_test = np.zeros((num_subs, num_forms))
    overall_train_mat = np.zeros((num_forms, 1))
    overall_test_mat = np.zeros((num_forms, 1))

    if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size)):
        os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size))

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # For each subject LOSO
    for i in range(num_subs):
        print('\n\nTraining subject: {}'.format(i+1))
        train_data = TaijiData(data_dir=args.data_root, subject=i+1, split='train')
        test_data = TaijiData(data_dir =args.data_root, subject=i+1, split='test')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        model = MLP(input_dim=train_data.data_dim, hidden_dim=1024, output_dim=num_forms).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Train + test the model
        model, train_losses, per_epoch_train_acc, train_preds, train_targets \
                        = train(model, train_loader, optimizer, criterion, args.num_epochs, args.log_interval, device)
        test_loss, test_acc, test_pred, test_targets = test(model, test_loader, device, criterion)

        # Print accs to three decimal places
        sub_train_acc[i] = per_epoch_train_acc[-1]
        sub_test_acc[i] = test_acc
        print(f'Subject {i+1} Train Accuracy: {per_epoch_train_acc[-1]*100:.3f}')
        print(f'Subject {i+1} Test Accuracy: {test_acc*100:.3f}')
        
        # Save all stats (you can save the model if you choose to)
        if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size, 'stats')):
            os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size, 'stats'))

        sub_file = os.path.join(args.save_dir, 'Taiji',args.fp_size, 'stats', 'sub_{}.npz'.format(i+1))
        classes_train, conf_mat_train = get_stats(train_preds, train_targets, num_forms)
        classes_test, conf_mat_test = get_stats(test_pred, test_targets, num_forms)
        sub_class_train[i, :] = classes_train
        sub_class_test[i, :] = classes_test
        overall_train_mat = overall_train_mat + (1/num_subs) * conf_mat_train 
        overall_test_mat = overall_test_mat + (1/num_subs) * conf_mat_test 
        np.savez(sub_file, train_losses=train_losses, per_epoch_acc=per_epoch_train_acc, test_acc=test_acc,
                 conf_mat_train=conf_mat_train, conf_mat_test=conf_mat_test)

        # Note: the code does not save the model, but you may choose to do so with the arg.save_model flag
        
    # Save overall stats
    overall_train_acc = np.mean(sub_train_acc)
    overall_test_acc = np.mean(sub_test_acc)
    print(f"\n\nOverall Train Accuracy: {overall_train_acc:.3f}")
    print(f"Overall Test Accuracy: {overall_test_acc:.3f}")

    overall_file_name = os.path.join(args.save_dir, 'Taiji', args.fp_size, 'stats', 'overall.npz')
    np.savez(overall_file_name, sub_train_acc = sub_train_acc, sub_class_train=sub_class_train,
             sub_test_acc=sub_test_acc, sub_class_test=sub_class_test, overall_train_mat=overall_train_mat, overall_test_mat=overall_test_mat)


def evaluate_model(model: nn.Module, args: Any):
    num_classes = 17
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    if args.model_type == 'CNN2':
        preprocess = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ]
    elif args.model_type == 'Resnet':
        preprocess = [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    transform = transforms.Compose(preprocess)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # model = CNN2(input_channels=1, img_size=args.img_size, num_classes=num_classes).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if args.model_path and not args.eval_ckpt:
        model.load_state_dict( torch.load(args.model_path) )
    elif args.model_path and args.eval_ckpt:
        state_dict = torch.load(args.model_path)['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        print('No model .pth or .ckpt file found!')
    
    test_loss, test_acc, test_preds, test_targets = test(model, test_loader, device, criterion)
    classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)
    print(f'Test accuracy: {test_acc*100:.3f}')



# ------------------------------------- Extra Credit Implementation ---------------------------------------------------
def resnet_main(args: Any):
    """
    Main function for training and testing the Resnet Network.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_classes = 17
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load the wallpaper dataset
    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: Augment the training data given the transforms in the assignment description.
    preprocess = [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    augmentation = [
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(1,2)),
    ]
    augmentation = preprocess + augmentation


    # Compose the transforms that will be applied to the images. Feel free to adjust this.
    transform = transforms.Compose(preprocess)
    augment = transforms.Compose(augmentation)
    train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    aug_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=augment)
    datasets = [train_dataset, aug_dataset]
    train_dataset = ConcatDataset(datasets)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images.")
    # Initialize the model, optimizer, and loss function
    model = Resnet(pretrained=True).to(device)

    # Freeze backbone for transfer learning
    # and leave a few unfrozen layers for finetuning
    for child in list( model.children() )[0][:-2][-1][:-4]:
        for param in child.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train + test the model
    model, per_epoch_loss, per_epoch_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, args.num_epochs, 
                                                                             args.log_interval, device, args.log_dir )
    test_loss, test_acc, test_preds, test_targets = test(model, test_loader, device, criterion)

    # Get stats 
    classes_train, overall_train_mat = get_stats(train_preds, train_targets, num_classes)
    classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)


    print(f'\n\nTrain accuracy: {per_epoch_acc[-1]*100:.3f}')
    print(f'Test accuracy: {test_acc*100:.3f}')

    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats')):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats'))
    overall_file_name = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats', 'overall.npz')

    np.savez(overall_file_name, classes_train=classes_train, overall_train_mat=overall_train_mat, 
                classes_test=classes_test, overall_test_mat=overall_test_mat, 
                per_epoch_loss=per_epoch_loss, per_epoch_acc=per_epoch_acc, 
                test_loss=test_loss, test_acc=test_acc)

    # Note: The code does not save the model but you may do so if you choose with the args.save_model flag.
    if args.save_model:
        model_dir = os.path.join( args.save_dir, 'cnn.pth' )
        torch.save(model.state_dict(), model_dir) 


def densenet_main(args: Any):
    """
    Main function for training and testing the Densenet Network.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_classes = 17
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load the wallpaper dataset
    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: Augment the training data given the transforms in the assignment description.
    preprocess = [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    augmentation = [
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(1,2)),
    ]
    augmentation = preprocess + augmentation


    # Compose the transforms that will be applied to the images. Feel free to adjust this.
    transform = transforms.Compose(preprocess)
    augment = transforms.Compose(augmentation)
    train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    aug_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=augment)
    datasets = [train_dataset, aug_dataset]
    train_dataset = ConcatDataset(datasets)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images.")
    # Initialize the model, optimizer, and loss function
    model = Densenet(pretrained=True).to(device)

    # Freeze backbone for transfer learning
    # and leave a few unfrozen layers for finetuning
    for child in list(model.children())[:-1][0][0][:-1]:
        for param in child.parameters():
            param.requires_grad_(False)
    
    last_dense_block = list(model.children())[:-1][0][0][:-1][-1]
    freeze_layers = [
        last_dense_block.denselayer1, last_dense_block.denselayer2, 
        last_dense_block.denselayer3, last_dense_block.denselayer4,
        last_dense_block.denselayer5, last_dense_block.denselayer6,
        last_dense_block.denselayer7, last_dense_block.denselayer8, 
        last_dense_block.denselayer9, last_dense_block.denselayer10,
        last_dense_block.denselayer11, last_dense_block.denselayer12,
        last_dense_block.denselayer13,
        # Less tuning:
        last_dense_block.denselayer14, last_dense_block.denselayer15
    ]
    for layer in freeze_layers:
        for param in list( layer.parameters() ):
            param.requires_grad_(False)


    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train + test the model
    model, per_epoch_loss, per_epoch_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, args.num_epochs, 
                                                                             args.log_interval, device, args.log_dir )
    test_loss, test_acc, test_preds, test_targets = test(model, test_loader, device, criterion)

    # Get stats 
    classes_train, overall_train_mat = get_stats(train_preds, train_targets, num_classes)
    classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)


    print(f'\n\nTrain accuracy: {per_epoch_acc[-1]*100:.3f}')
    print(f'Test accuracy: {test_acc*100:.3f}')

    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats')):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats'))
    overall_file_name = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats', 'overall.npz')

    np.savez(overall_file_name, classes_train=classes_train, overall_train_mat=overall_train_mat, 
                classes_test=classes_test, overall_test_mat=overall_test_mat, 
                per_epoch_loss=per_epoch_loss, per_epoch_acc=per_epoch_acc, 
                test_loss=test_loss, test_acc=test_acc)

    # Note: The code does not save the model but you may do so if you choose with the args.save_model flag.
    if args.save_model:
        model_dir = os.path.join( args.save_dir, 'cnn.pth' )
        torch.save(model.state_dict(), model_dir) 



if __name__ == '__main__':
    args = arg_parse()

    if args.test_model:
        if args.model_type == 'CNN2':
            model = CNN2(input_channels=1, img_size=args.img_size, num_classes=17)
        elif args.model_type == 'Resnet':
            model = Resnet()
        else:
            raise ValueError('{} model not supported, please try Resnet/CNN2'.format(args.model_type))
        evaluate_model(model, args)
    elif args.resnet:
        resnet_main(args)
        visualize(args, dataset='Wallpaper')
        plot_training_curve(args)
    elif args.densenet:
        densenet_main(args)
        visualize(args, dataset='Wallpaper')
        plot_training_curve(args)
    else:
        if args.dataset == 'Wallpaper':
            if args.train:
                wallpaper_main(args)
            visualize(args, dataset='Wallpaper')
            plot_training_curve(args)
        else: 
            if args.train:
                taiji_main(args)
            visualize(args, dataset='Taiji')
            plot_training_curve(args)

