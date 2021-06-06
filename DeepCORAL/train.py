import torch
import torch.nn as nn
import time
from dl_models import TransferNet, TransferModel, FinetuneModel
from utils import load_data, load_feature, data_preprocessing


def finetune(model, dataloaders, optimizer):
    since = time.time()
    best_acc = 0
    stop = 0
    for epoch in range(0, n_epoch):
        stop += 1
        # You can uncomment this line for scheduling learning rate
        # lr_schedule(optimizer, epoch)
        for phase in ['src', 'val', 'tar']:
            if phase == 'src':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(correct) / len(dataloaders[phase].dataset)  # correct.double() --> float(correct)
            print(f'Epoch: [{epoch:02d}/{n_epoch:02d}]---{phase}, loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                stop = 0
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'saved_models/model.pkl')
        if stop >= early_stop:
            break
        print()
    
    time_pass = time.time() - since
    print(f'Training complete in {time_pass // 60:.0f}m {time_pass % 60:.0f}s')


def finetune_ef(model, dataloaders, optimizer, lr, name="AA"):
    since = time.time()
    best_acc = 0
    stop = 0
    learning_rate = lr
    best_epoch = 0
    for epoch in range(0, n_epoch):
        stop += 1
        # You can uncomment this line for scheduling learning rate
        # lr_schedule(optimizer, epoch)
        for phase in ['src', 'val']:
            if phase == 'src':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(correct) / len(dataloaders[phase].dataset)  # correct.double() --> float(correct)
            print(f'Epoch: [{epoch:02d}/{n_epoch:02d}]---{phase}, loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                stop = 0
                best_acc = epoch_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'saved_models/model_%s.pkl' % name)
        if stop >= lr_decay_num and learning_rate >= 1e-5:
            print("Learning rate decay, original lr:", learning_rate, "current lr:", learning_rate / 2)
            learning_rate /= 2
            optimizer.param_groups[0]['lr'] = learning_rate
            stop = 0
        if stop >= early_stop:
            break
        print()
    
    time_pass = time.time() - since
    print(f'Training complete in {time_pass // 60:.0f}m {time_pass % 60:.0f}s')
    print("Best acc:", best_acc, "Best epoch:", best_epoch)


def train(dataloaders, model, optimizer, lr, name="AA"):
    since = time.time()
    source_loader, target_train_loader, target_test_loader = dataloaders['src'], dataloaders['val'], dataloaders['tar']
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    learning_rate = lr
    best_epoch = 0
    best_acc = 0
    stop = 0
    lamb = 10  # weight for transfer loss, it is a hyper-parameter that needs to be tuned
    n_batch = min(len_source_loader, len_target_loader)
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        model.train()
        for (src, tar) in zip(source_loader, target_train_loader):
            data_source, label_source = src
            data_target, _ = tar
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
            
            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred,
                                 torch.tensor(label_source, dtype=torch.long))
            loss = clf_loss + lamb * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() + train_loss_clf
            train_loss_transfer = transfer_loss.detach().item() + train_loss_transfer
            train_loss_total = loss.detach().item() + train_loss_total
        acc = test(model, target_test_loader)
        print(
            f'Epoch: [{e:2d}/{n_epoch}], cls_loss: {train_loss_clf / n_batch:.4f}, transfer_loss: {train_loss_transfer / n_batch:.4f}, total_Loss: {train_loss_total / n_batch:.4f}, acc: {acc:.4f}')
        if best_acc < acc:
            best_acc = acc
            best_epoch = e
            torch.save(model.state_dict(), 'saved_models/trans_model_%s.pkl' % name)
            stop = 0
        if stop >= lr_decay_num and learning_rate >= 1e-5:
            print("Learning rate decay, original lr:", learning_rate, "current lr:", learning_rate / 2)
            learning_rate /= 2
            optimizer.param_groups[0]['lr'] = learning_rate
            stop = 0
        if stop >= early_stop:
            break
    time_pass = time.time() - since
    print(f'Training complete in {time_pass // 60:.0f}m {time_pass % 60:.0f}s')
    print("Best acc:", best_acc, "Best epoch:", best_epoch)


def test(model, target_test_loader):
    model.eval()
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = float(correct) / len_target_dataset  # correct.double() --> float(correct)
    return acc


def fine_tune_model(dataloaders):
    """Fine Tune Model"""
    param_group = []
    learning_rate = 0.0001
    momentum = 5e-4
    # build model
    model = TransferModel().cuda()
    RAND_TENSOR = torch.randn(1, 3, 224, 224).cuda()
    output = model(RAND_TENSOR)
    print(output)
    print(output.shape)
    for k, v in model.named_parameters():
        """learning rate for the FC layer is 10 times of other layers, which is a common trick."""
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 10}]
    optimizer = torch.optim.SGD(param_group, momentum=momentum)
    # fine-tune model
    finetune(model, dataloaders, optimizer)
    # test model
    model.load_state_dict(torch.load('saved_models/model.pkl'))
    acc_test = test(model, dataloaders['tar'])
    print(f'Test accuracy: {acc_test}')


def fine_tune_model_ef(src_loader, tar_loader, name="AA"):
    """Fine Tune Model"""
    # set model
    dataloaders = {'src': src_loader,
                   'val': tar_loader,
                   'tar': tar_loader}
    param_group = []
    learning_rate = 0.001
    momentum = 5e-4
    # build model
    model = FinetuneModel().cuda()
    for k, v in model.named_parameters():
        """learning rate for the FC layer is 10 times of other layers, which is a common trick."""
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 10}]
    # optimizer = torch.optim.SGD(param_group, momentum=momentum)
    optimizer = torch.optim.Adam(param_group, lr=learning_rate)
    # fine-tune model
    finetune_ef(model, dataloaders, optimizer, learning_rate, name=name)
    # test model
    model.load_state_dict(torch.load('saved_models/model_%s.pkl' % name))
    acc_test = test(model, dataloaders['tar'])
    print(f'Test accuracy: {acc_test}')


def domain_adaptation(src_loader, tar_loader, name="AA"):
    """Transfer Model"""
    dataloaders = {'src': src_loader,
                   'val': tar_loader,
                   'tar': tar_loader}
    # build model
    transfer_loss = 'coral'
    learning_rate = 0.0001
    transfer_model = TransferNet(n_class, transfer_loss=transfer_loss, base_net='base_fc').cuda()
    # optimizer = torch.optim.SGD([
    #     {'params': transfer_model.base_network.parameters()},
    #     {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * learning_rate},
    #     {'params': transfer_model.classifier_layer.parameters(), 'lr': 10 * learning_rate},
    # ], lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(params=[
        {'params': transfer_model.base_network.parameters()},
        {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * learning_rate},
        {'params': transfer_model.classifier_layer.parameters(), 'lr': 10 * learning_rate},
    ], lr=learning_rate)
    # train
    train(dataloaders, transfer_model, optimizer, lr=learning_rate, name=name)
    # test
    transfer_model.load_state_dict(torch.load('saved_models/trans_model_%s.pkl' % name))
    acc_test = test(transfer_model, dataloaders['tar'])
    print(f'Test accuracy: {acc_test}')


if __name__ == '__main__':
    batch_size = 64
    n_class = 65
    n_epoch = 60        # 500
    early_stop = 20     # 40
    lr_decay_num = 10       # 20
    criterion = nn.CrossEntropyLoss()
    # prepare data loader
    sd_features_AA, sd_features_CC, sd_features_PP, \
    td_features_AR, td_features_CR, td_features_PR, \
    sd_labels_AA, sd_labels_CC, sd_labels_PP, \
    td_labels_AR, td_labels_CR, td_labels_PR = data_preprocessing()
    AA_loader, AR_loader = load_feature(Xs=sd_features_AA, Ys=sd_labels_AA, Xt=td_features_AR, Yt=td_labels_AR,
                                        batch_size=batch_size)
    CC_loader, CR_loader = load_feature(Xs=sd_features_CC, Ys=sd_labels_CC, Xt=td_features_CR, Yt=td_labels_CR,
                                        batch_size=batch_size)
    PP_loader, PR_loader = load_feature(Xs=sd_features_PP, Ys=sd_labels_PP, Xt=td_features_PR, Yt=td_labels_PR,
                                        batch_size=batch_size)
    print(f'Source data number (AA): {len(AA_loader.dataset)}')
    print(f'Target data number (AR): {len(AR_loader.dataset)}')
    # fine-tune model
    print("\n \n Fine tune Model: AA AR")
    fine_tune_model_ef(src_loader=AA_loader, tar_loader=AR_loader, name="AA")
    print("\n \n Fine tune Model: CC CR")
    fine_tune_model_ef(src_loader=CC_loader, tar_loader=CR_loader, name="CC")
    print("\n \n Fine tune Model: PP PR")
    fine_tune_model_ef(src_loader=PP_loader, tar_loader=PR_loader, name="PP")
    # Deep CORAL model
    print("\n \n Domain Adaptation Model: AA AR")
    domain_adaptation(src_loader=AA_loader, tar_loader=AR_loader, name="AA")
    print("\n \n Domain Adaptation Model: CC CR")
    domain_adaptation(src_loader=CC_loader, tar_loader=CR_loader, name="CC")
    print("\n \n Domain Adaptation Model: PP PR")
    domain_adaptation(src_loader=PP_loader, tar_loader=PR_loader, name="PP")
    
