from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

from .models import TCN, save_model
from .utils import one_hot, load_speech_data


def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_7'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_7'), flush_secs=1)

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = TCN().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th')))
    loss = torch.nn.NLLLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    batch = 1
    train_data = load_speech_data('<path to train data>', transform=one_hot, batch_size=batch)
    valid_data = load_speech_data('<path to test data>', transform=one_hot, batch_size=batch)

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})

    lr_finder = FastaiLRFinder()
    to_save = {'model': model, 'optimizer': optimizer}
    with lr_finder.attach(trainer, to_save, diverge_th=1.5) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(train_data)

    trainer.run(train_data, max_epochs=10)

    evaluator = create_supervised_evaluator(model, metrics={"loss": Loss(torch.nn.NLLLoss())},
                                            device=device)
    evaluator.run(valid_data)

    print(evaluator.state.metrics)
    lr_finder.plot()
    global_step = 0
    for iterations in range(args.n_iterations):
        model.train()
        for batches in train_data:
            batch_data = batches[:, :, :-1].to(device)
            batch_label = batches.argmax(dim=1).to(device)
            o = model(batch_data)
            loss_val = loss(o, batch_label)
            train_logger.add_scalar('train/loss', loss_val, global_step=global_step)
            global_step += 1
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        save_model(model)

        model.eval()
        for data in valid_data:
            batch_data_valid = data[:, :, :-1].to(device)
            batch_label_valid = data.argmax(dim=1).to(device)
            o = model(batch_data_valid)
            loss_valid = loss(o, batch_label_valid)
            valid_logger.add_scalar('valid/loss', loss_valid, global_step=global_step)

        print("iterations: ", iterations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--n_iterations', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
