def train(dataloader, model, loss_fn, optimizer):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    model.train()

    for batch, (y, X) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.flatten().long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 10 == 0:
        #    loss = loss.item()
        #    current = batch * len(X)

        #    if batch == 10:
        #        with open("modelLog.txt", "a") as modelLog:
        #            modelLog.write(f"loss: {loss:>7f}  [{current}/{size}]")
        #        print(f"loss: {loss:>7f}  [{current}/{size}]")
        #    else:
        #        with open("modelLog.txt","a") as modelLog:
        #            modelLog.write(f"loss: {loss:>7f}  [{current}/{size}]")
        #        print(f"loss: {loss:>7f}  [{current}/{size}]")


def test(dataloader, model, loss_fn):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0, 0, 0
    track = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if track == 0:
                preds = pred
            else:
                preds = torch.cat((preds, pred), 0)
            track += 1
            test_loss += loss_fn(pred, y.flatten().long()).item()
            correct += (pred.argmax(1) == y.flatten().long()).type(torch.float).sum().item()
            true_positive += ((pred.argmax(1) == y.flatten().long()) & (y.flatten().long() == 1)).type(
                torch.float).sum().item()
            true_negative += ((pred.argmax(1) == y.flatten().long()) & (y.flatten().long() == 0)).type(
                torch.float).sum().item()
            false_positive += ((pred.argmax(1) != y.flatten().long()) & (y.flatten().long() == 0)).type(
                torch.float).sum().item()
            false_negative += ((pred.argmax(1) != y.flatten().long()) & (y.flatten().long() == 1)).type(
                torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    with open("modelLog.txt", "a") as modelLog:
        modelLog.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        modelLog.write(f"true positive: {true_positive}\ntrue negative: {true_negative}\nfalse positive: "
                       f"{false_positive}\nfalse "
                       f"negative: {false_negative}\n")
        modelLog.write(f"f1 score: {true_positive / (true_positive + 1 / 2 * (false_positive + false_negative))}")
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"true positive: {true_positive}\ntrue negative: {true_negative}\nfalse positive: {false_positive}\nfalse "
          f"negative: {false_negative}\n")
    print(f"f1 score: {true_positive / (true_positive + 1 / 2 * (false_positive + false_negative))}")

    return preds

def validate(model, test_dataset, test_dataloader):
    from torch import nn
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot

    loss_fn = nn.CrossEntropyLoss()

    preds = test(test_dataloader, model, loss_fn)

    genprob = nn.Sigmoid()
    predplot = genprob(preds[:, 1])
    lr_fpr, lr_tpr, _ = roc_curve(test_dataset.tensors[1].flatten(), predplot)

    pyplot.plot(lr_fpr, lr_tpr)
    pyplot.savefig("RUC.png")

def network(train_dataloader,test_dataloader,epochs):
    import torch
    from torch import nn

    #labs = train_dataloader.dataset.labels
    #pos_weight = 1-sum(labs==1)/(sum(labs==1)+sum(labs==0))
    #neg_weight = 1-sum(labs == 0) / (sum(labs == 1) + sum(labs == 0))

    #weight_vector = torch.tensor([neg_weight,pos_weight])

    with open("modelLog.txt", "w") as modelLog:
        modelLog.write(f"Initializing Run")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")

    #initialize newtork hyperparameters
    #currently arbitrary, will be determined with tuning script for future versions
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.dropout = nn.Dropout(.4)
            #self.hidden1 = nn.Linear(300,200)
            self.hidden1 = nn.Linear(300,200)
            self.hidden2 = nn.Linear(200,150)
            self.hidden3 = nn.Linear(150,2)

            self.activate = nn.ReLU()
            #self.activate = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden1(x)
            x = self.activate(x)
            x = self.dropout(x)
            x = self.hidden2(x)
            x = self.activate(x)
            x = self.dropout(x)
            x = self.hidden3(x)
            return x

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.HingeEmbeddingLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    for t in range(epochs):
        with open("modelLog.txt", "a") as modelLog:
            modelLog.write(f"Epoch {t+1}\n-------------------------------")
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader,model,loss_fn)
        scheduler.step()
    print("Done!")

    return model

#FUNCTION UNDER DEVELOPMENT
def network_tune(train_dataloader,test_dataloader):
    class NeuralNetwork(nn.Module):
        def __init__(self,n_neurons=10):
            super(NeuralNetwork, self).__init__()
            self.hidden = nn.Linear(300,n_neurons)
            #self.hidden1 = nn.Linear(300,200)
            #self.hidden2 = nn.Linear(200,150)
            #self.hidden3 = nn.Linear(150,2)

            self.activate = nn.ReLU()

        def forward(self, x):
            x = self.hidden1(x)
            x = self.activate(x)
            x = self.hidden2(x)
            x = self.activate(x)
            x = self.hidden3(x)
            return x

    model = NeuralNetClassifier(
        module=NeuralNetwork
    )











