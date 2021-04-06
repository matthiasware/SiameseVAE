import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import torch.nn.functional as F


def down_knn(data_loader, model, device, n_neighbors=5):
    model.eval()
    model = model.to(device)
    #
    features = []
    targets = []
    #
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='Down knn valid') as p_bar:
            for data, target in data_loader:
                feature = model(data.to(device))
                features.append(feature.detach().cpu().numpy())
                targets.append(target.cpu().numpy())
                p_bar.update()
            #
            x = np.concatenate(features)
            y = np.concatenate(targets)
            #
            neigh = KNeighborsClassifier(n_neighbors=n_neighbors,
                                         algorithm='brute', n_jobs=8)
            #
            neigh.fit(x, y)
            score = neigh.score(x, y) * 100
            p_bar.set_postfix({"acc": score})
    return score


def down_train_linear(
        model,
        classifier,
        data_loader,
        optimizer,
        device,
        num_epochs):
    model.eval()
    #
    model = model.to(device)
    classifier = classifier.to(device)
    #
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #
    p_bar = tqdm(range(num_epochs), desc='Down lin train')
    for epoch in p_bar:

        losses, acc, step, total = 0., 0., 0., 0.
        for data, target in data_loader:
            #
            data = data.to(device)
            target = target.to(device)
            #
            with torch.no_grad():
                z = model(data)
            logits = classifier(z)
            #
            optimizer.zero_grad()
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            #
            losses += loss.item()
            #
            pred = F.softmax(logits, dim=-1).max(-1)[1]
            acc += pred.eq(target).sum().item()
            step += 1
            total += target.size(0)

        loss_avg = losses / step
        acc_avg = acc / total * 100

        p_bar.set_postfix(
            {'loss': loss_avg, 'acc': acc_avg})
    return loss_avg, acc_avg


def down_valid_linear(
        model,
        classifier,
        data_loader,
        device):
    #
    model.eval()
    classifier.eval()
    #
    model = model.to(device)
    classifier = classifier.to(device)
    #
    with torch.no_grad():
        acc, total = 0., 0.
        p_bar = tqdm(data_loader, desc='Down lin valid')
        for data, target in p_bar:
            data = data.to(device)
            target = target.to(device)
            z = model(data)
            logits = classifier(z)
            #
            pred = F.softmax(logits, dim=-1).max(-1)[1]
            acc += pred.eq(target).sum().item()
            #
            total += target.size(0)

            p_bar.set_postfix(
                {'acc': acc / total * 100})

    acc_avg = acc / total * 100
    return acc_avg


def std_cov_valid(data_loader, model, device):
    model.eval()
    #
    outs = []
    targets = []
    #
    with tqdm(total=len(data_loader), desc='Down STD') as p_bar:
        with torch.no_grad():
            for data, target in data_loader:
                out = model(data.to(device))
                outs.append(out.detach().cpu().numpy())
                targets.append(target.cpu().numpy())
                #
                p_bar.update()

        x = np.concatenate(outs)
        #
        norms = np.linalg.norm(x, axis=1)
        z_bars = x / norms[:, None]
        std = z_bars.std(axis=0).mean()

        cov = np.cov(x.T)
        p_bar.set_postfix({"STD": std})
    return std, cov
