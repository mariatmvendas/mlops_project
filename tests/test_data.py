import sys
from mlops_project.train import train_dataloader_satellite

def test_train_dataloader_satellite():
    train_images, train_targets = train_dataloader_satellite()
    sys.stdout.write(f"Number of train images: {len(train_images)}\n")
    assert len(train_images) == 4504
    assert len(train_targets) == 4504


    # for dataset in [train, test]:
    #     for x, y in dataset:
    #         assert x.shape == (1, 28, 28)
    #         assert y in range(10)
    # train_targets = torch.unique(train.tensors[1])
    # assert (train_targets == torch.arange(0,10)).all()
    # test_targets = torch.unique(test.tensors[1])
    # assert (test_targets == torch.arange(0,10)).all()
