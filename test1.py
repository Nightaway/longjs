import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # print(m.weight.data.size())
        if len(m.weight.data.size()) == 2:
            for i in range(m.weight.data.size()[0]):
                for j in range(m.weight.data.size()[1]):
                    m.weight.data[i][j] = 0.00005
        else:
            for i in range(m.weight.data.size()[0]):
                    m.weight.data[i] = 0.00005
        for i in range(m.bias.data.size()[0]):
                    m.bias.data[i] = 0.00005
        # print(m.bias.data)

model.apply(weights_init)

for param in model.parameters():
    print(param.data)

x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(x)

for i in range(0, 10):
    y_ = model(x)
    print(y_)

    loss = y_ - 1
    print(loss)

    loss.backward()

    print('------')
    for f in model.parameters():
        print(f.grad.data)
    print('------')

    learning_rate = 1
    for f in model.parameters():
        f.data.sub_(learning_rate * f.grad.data)
        f.grad.data.zero_()
