import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
from datetime import datetime
import json

from albumentations import (
    Compose,
    Normalize,
    Resize
)

import config
from dataset import RobotSegDataset
from metrics import LossBinary, LossMulti, validation_binary, validation_multi
from preprocess_data import trainval_split


def main():
    # check cuda available
    assert torch.cuda.is_available() == True

    # when the input dimension doesnot change, add this flag to speed up
    cudnn.benchmark = True

    num_classes = config.num_classes[config.problem_type]
    # input are RGB images in size 3 * h * w
    # output are binary
    model = config.model(3, num_classes)
    # data parallel
    model = nn.DataParallel(model, device_ids=config.device_ids).cuda()
    # loss function
    if num_classes == 2:
        loss = LossBinary(jaccard_weight=config.jaccard_weight)
        valid_metric = validation_binary
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=config.jaccard_weight)
        valid_metric = validation_multi


    # train/valid filenmaes
    train_filenames, valid_filenames = trainval_split(config.fold)
    print('num of train / validation files = {} / {}'.format(len(train_filenames), len(valid_filenames)))


    # trainset transform
    train_transform = Compose([
            Resize(height=config.train_height, width=config.train_width, p=1),
            Normalize(p=1)
        ], p=1)

    # validset transform
    valid_transform = Compose([
            Resize(height=config.train_height, width=config.train_width, p=1),
            Normalize(p=1)
        ], p=1)

    # train dataloader
    train_loader = DataLoader(
            dataset=RobotSegDataset(train_filenames, transform=train_transform),
            shuffle=True,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True
        )
    # valid dataloader
    valid_loader = DataLoader(
            dataset=RobotSegDataset(valid_filenames, transform=valid_transform),
            shuffle=True,
            num_workers=config.num_workers,
            batch_size=len(config.device_ids), # in valid time use one img for each dataset
            pin_memory=True
        )

    train(
        model=model,
        loss_func=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_metric=valid_metric,
        fold=config.fold,
        num_classes=num_classes
    )

def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()
        
def save_model(model, model_path, epoch, step):
    print('Saving model...')
    torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'step': step,
        }, str(model_path))
    print('Finished.')



def train(model, loss_func, train_loader, valid_loader, valid_metric, fold=None,
          num_classes=2):
    lr = config.lr

    # TODO: adaptive lr
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load model/train from scratch
    config.model_dir.mkdir(exist_ok=True, parents=True)
    model_path = Path(config.model_dir) / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model [{}] from epoch {}, step {:,}'.format(config.model.__name__, epoch, step))
    else:
        epoch = 1
        step = 0

    # record {#} batches
    report_each = 10

    # write train log to file
    log_path = config.model_dir / 'train_{fold}.log'.format(fold=fold)
    log = log_path.open('a+', encoding='utf8')

    valid_losses = []
    for ep in range(epoch, config.epoches + 1):
        # set mode to train
        model.train()

        # init progress bar for each epoch
        tq = tqdm.tqdm(total=(len(train_loader) * config.batch_size))
        tq.set_description('Train [{}] epoch {} lr {}'.format(config.model.__name__, ep, lr))
        
        # record losses for each batch
        losses = []
        try:
            mean_loss = 0
            # TODO: add optical flow
            for i, (inputs, targets, optflow) in enumerate(train_loader):
                # no grad for targets
                inputs = inputs.cuda(non_blocking=True)
                with torch.no_grad():
                    targets = targets.cuda(non_blocking=True)

                # # clear gradients
                optimizer.zero_grad()
                outputs, _ = model(inputs, optflow)
                loss = loss_func(outputs, targets)

                loss.backward()
                optimizer.step()
                # update step
                step += 1

                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])

                # update progress bar
                tq.update(config.batch_size)
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            
            valid_metrics = valid_metric(model, loss_func, valid_loader, num_classes)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            write_event(log, step, **valid_metrics)

            # close progress bar
            tq.close()

        except KeyboardInterrupt:
            tq.close()
            # save model
            print("Ctrl+C, training paused")
            save_model(model, model_path, ep + 1, step)
            return


    # save model after training
    save_model(model, model_path, config.epoches + 1, step)




if __name__ == '__main__':
    main()
