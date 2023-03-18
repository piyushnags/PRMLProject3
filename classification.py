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
          log_interval, device, log_dir=None, scheduler=None):
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

        if scheduler is not None:
            scheduler.step()

        print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch+1, train_loss, train_acc))
        if (epoch+1) % log_interval == 0:            
            # Save Checkpoints every log_interval epochs
            if log_dir is not None:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                ckpt_path = os.path.join(log_dir, 'ckpt_{}.ckpt'.format(epoch+1))
                
                if scheduler is not None:
                    scheduler_val = scheduler.state_dict()
                else:
                    scheduler_val = None
                
                torch.save({
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "epoch":epoch,
                    "per_epoch_loss":per_epoch_loss,
                    "per_epoch_acc":per_epoch_acc,
                    "scheduler_state_dict":scheduler_val
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
    elif args.model_type == 'Resnet' or args.model_type == 'Densenet' or args.model_type == 'Mobilenet':
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


def resume_training(args):
    if not os.path.exists(args.model_path):
        raise ValueError('Path {} does not exist!!'.format(args.model_path))
    ckpt = torch.load(args.model_path)

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if args.model_type == 'Resnet':
        model = Resnet().to(device)
        # Need to freeze model layers before loading optimizer state dict
        for child in list( model.children() )[0][:-2][-1][:-4]:
            for param in child.parameters():
                param.requires_grad_(False)
    elif args.model_type == 'Densenet':
        model = Densenet().to(device)
    elif args.model_type == 'Mobilenet':
        model = Mobilenet().to(device)
    else:
        raise ValueError('Invalid model type!')

    model.load_state_dict( ckpt['model_state_dict'] )
    
    
    # optimizer = torch.optim.Adam( [p for p in model.parameters() if p.requires_grad], lr=args.lr )
    
    # Google's Training Recipe
    optimizer = torch.optim.RMSprop( 
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.lr, 
        momentum=0.9,
        weight_decay=1e-5,
        eps=0.0316,
        alpha=0.9
        )
    optimizer.load_state_dict( ckpt['optimizer_state_dict'] )
    
    
    scheduler_state_dict = ckpt['scheduler_state_dict']
    scheduler = None
    if scheduler_state_dict is not None:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.973)
        scheduler.load_state_dict(scheduler_state_dict)
    start_epoch = ckpt['epoch'] + 1
    end_epoch = start_epoch + args.num_epochs
    per_epoch_loss = ckpt['per_epoch_loss']
    per_epoch_acc = ckpt['per_epoch_acc']

    # Load the wallpaper dataset
    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.model_type == 'CNN2':
        preprocess = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ]
    elif args.model_type == 'Resnet' or args.model_type == 'Densenet' or args.model_type == 'Mobilenet':
        preprocess = [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(start_epoch, end_epoch+1):
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
            if epoch == args.num_epochs - 1:
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        per_epoch_acc.append(train_acc)

        if scheduler is not None:
            scheduler.step()

        print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch, train_loss, train_acc))
        if (epoch) % args.log_interval == 0:            
            # Save Checkpoints every log_interval epochs
            if args.log_dir is not None:
                if not os.path.exists(args.log_dir):
                    os.makedirs(args.log_dir)
                ckpt_path = os.path.join(args.log_dir, 'ckpt_{}.ckpt'.format(epoch))
                
                if scheduler is not None:
                    scheduler_val = scheduler.state_dict()
                else:
                    scheduler_val = None
                
                torch.save({
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "epoch":epoch - 1,
                    "per_epoch_loss":per_epoch_loss,
                    "per_epoch_acc":per_epoch_acc,
                    "scheduler_state_dict":scheduler_val
                }, ckpt_path)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return model, np.array(per_epoch_loss), np.array(per_epoch_acc), preds, targets        


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
    for child in list(model.children())[:-2][0][0][:-2]:
        for param in child.parameters():
            param.requires_grad_(False)
    
    last_dense_block = list(model.children())[:-2][0][0][:-1][-1]
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


def mobilenet_main(args: Any):
    """
    Main function for training and testing the MobileNetv3 (Small) Network.
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
    model = Mobilenet().to(device)
    # for param in ( list( model.children() )[:-1][0][0][:-3] ).parameters():
    #     param.requires_grad_(False)

    # optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 1e-6)

    # Google's Training Recipe
    optimizer = torch.optim.RMSprop( 
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.lr, 
        momentum=0.9,
        weight_decay=1e-5,
        eps=0.0316,
        alpha=0.9
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.973)
    criterion = nn.CrossEntropyLoss()

    # Train + test the model
    model, per_epoch_loss, per_epoch_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, args.num_epochs, 
                                                                             args.log_interval, device, args.log_dir, lr_scheduler )
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


def visualize_maps(args: Any, model: nn.Module):
    # Lists to store hook outputs
    activation = []
    tsne_input = [] 

    # Custom hook for storing feature maps 
    def get_activation():
        def hook(model, input, output):
            activation.append(output.detach()) 
        return hook

    # Custom hook for collecting fc layers output for 
    # TSNE visualization
    def get_fc():
        def hook(model, input, output):
            tsne_input.append(output.detach())
        return hook
    
    # Store feature maps after second convolution
    model.conv_layers[4].register_forward_hook(get_activation())

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    preprocess = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]

    transform = transforms.Compose(preprocess)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)

    # Get 1 image from each Wallpaper Group
    imgs = [ test_dataset[i][0] for i in range(0, 3400, 200) ]
    titles = [
        'CM', 'CMM', 'P1', 'P2', 'P3', 'P3M1', 'P4', 'P4G',
        'P4M', 'P6', 'P6M', 'P3M1', 'PG', 'PGG', 'PM', 'PMG',
        'PMM',
    ]
    
    # Create subplot of selected sample images
    fig = plt.figure( figsize=(15,15) )
    for (i, img), title in zip(enumerate(imgs), titles):
        ax = fig.add_subplot(4, 5, i+1)
        ax.axis('off')
        ax.set_title(title)
        ax.imshow(img.permute(1,2,0))
    plt.savefig(os.path.join(args.save_dir, 'sample_imgs.png'), dpi='figure')    
    
    # Get model
    model.to(device)
    model.load_state_dict( torch.load(args.model_path, map_location=device) )
    model.eval()

    input_batch = imgs[0].unsqueeze(0)
    for i in range(1, len(imgs)):
        input_batch = torch.cat( (input_batch, img.unsqueeze(0)) )

    with torch.no_grad():
        x = model(input_batch)
    
    # Plot feature maps after running the model
    fig = plt.figure( figsize=(15,15) )
    for i, title in zip(range(17), titles):
        feat_map = activation[0][i]
        ax = fig.add_subplot(4, 5, i+1)
        ax.axis('off')
        ax.set_title(title)
        ax.imshow(feat_map[0], cmap='gray')

    plt.savefig( os.path.join(args.save_dir, 'feat_maps.png'), dpi='figure' )
    # plt.show()

    # TSNE Visualization
    model.fc_1.register_forward_hook(get_fc())
    imgs = []
    for i in range(0, 3400, 200):
        for j in range(20):
            imgs.append( test_dataset[i+j][0] )
    
    input_batch = imgs[0].unsqueeze(0)
    for i in range(1, len(imgs)):
        input_batch = torch.cat( (input_batch, img.unsqueeze(0)) )
    
    with torch.no_grad():
        x = model(input_batch)

    pca = PCA(n_components=90)
    pca_results = pca.fit_transform(tsne_input[0])
    
    tsne = TSNE(n_components=3)
    tsne_results = tsne.fit_transform(pca_results)
    
    fig = plt.figure( figsize=(15,15) )
    ax = fig.add_subplot(projection='3d')
    c = np.arange(1, 290, 17)
    colors = [ c[i//20] for i in range(len(tsne_results.T[0])) ]
    scatter = ax.scatter(tsne_results.T[0], tsne_results.T[1], tsne_results.T[2], c=colors, marker='X')
    legend1 = ax.legend(*scatter.legend_elements(), title='Classes (idx)')
    ax.add_artist(legend1)
    plt.savefig( os.path.join(args.save_dir, 'TSNE.png'), dpi='figure' )
    plt.show()


if __name__ == '__main__':
    args = arg_parse()
    
    if args.test_model:
        if args.model_type == 'CNN2':
            model = CNN2(input_channels=1, img_size=args.img_size, num_classes=17)
        elif args.model_type == 'Resnet':
            model = Resnet()
        elif args.model_type == 'Densenet':
            model = Densenet()
        elif args.model_type == 'Mobilenet':
            model = Mobilenet()
        else:
            raise ValueError('{} model not supported, please try Resnet/CNN2'.format(args.model_type))
        evaluate_model(model, args)
        visualize(args, dataset='Wallpaper')
    
    elif args.maps:
        if args.device == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model = CNN2(input_channels=1, img_size=args.img_size, num_classes=17)
        model.load_state_dict( torch.load(args.model_path, map_location=device) )
        visualize_maps(args, model)
    
    elif args.resume_training:
        model, per_epoch_loss, per_epoch_acc, preds, targets = resume_training(args)
        model_dir = os.path.join( args.save_dir, 'cnn.pth' )
        torch.save(model.state_dict(), model_dir)
    elif args.resnet:
        resnet_main(args)
        visualize(args, dataset='Wallpaper')
        plot_training_curve(args)
    elif args.densenet:
        densenet_main(args)
        visualize(args, dataset='Wallpaper')
        plot_training_curve(args)
    elif args.mobilenet:
        mobilenet_main(args)
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