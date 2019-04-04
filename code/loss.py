import torch
import torch.nn as nn

def MultiLabelCrossEntropyLoss(output, target):
    loss, length = 0, target.size()[1] # output dims: 0 for sample, 1 for label, 2 for class    
    for i in range(length):
        loss += nn.CrossEntropyLoss()(output[:,i,:], target[:,i])

    return loss/length

if __name__ == "__main__":
    # multi-label one-class
    print('multi-label:')
    output = torch.FloatTensor([[5,-2,8],
                                [-6,4,-1]])
    target = torch.FloatTensor([[1,0,1],
                                [0,1,0]])
    loss = nn.BCEWithLogitsLoss()
    print(loss(output, target))

    # one-label multi-class
    print('\nmulti-class:')
    output = torch.FloatTensor([[-1,5,-2],
                                [-3,-2,6]])
    target = torch.tensor([1,2])
    loss = nn.CrossEntropyLoss()


    output = torch.FloatTensor([[-1,5,-2],
                                [-3,-2,6],
                                [-1,5,-2],
                                [-3,-2,6]])
    target = torch.tensor([1,2,1,2])
    loss = nn.CrossEntropyLoss()

    output = torch.FloatTensor([[-1,5,-2],
                                [3,-6,-3]])
    target = torch.tensor([1,0])
    loss = nn.CrossEntropyLoss()
    print(loss(output, target))

    # multi-label multi-class
    print('\nmulti-label multi-class:')
    output = torch.FloatTensor([
        [[-1,5,-2],
         [-1,5,-2],
         [8,0,-3]],

        [[3,-6,-3],
         [3,-6,-3],
         [-5,-1,5]]
         ])
    target = torch.tensor([[1,1,0],
                          [0,0,2]])

    loss = nn.CrossEntropyLoss()

    loss0 = loss(output[:,0,:], target[:,0])
    loss1 = loss(output[:,1,:], target[:,1])
    loss2 = loss(output[:,2,:], target[:,2])
    print('CrossEntropyLoss() on each label:')
    print(loss0)
    print(loss1)
    print(loss2)
    print('mean:')
    print((loss0+loss1+loss2)/3)

    print('my function:')
    cri = MultiLabelCrossEntropyLoss
    print(cri(output, target))

