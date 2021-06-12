from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

import utils
from models import TCN, save_model


def train(args):

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = TCN().to(device)
    loss = torch.nn.NLLLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    batch = 1
    train_data = load_speech_data('<path to train data>', transform=utils.one_hot, batch_size=batch)
    valid_data = load_speech_data('<path to test data>', transform=utils.one_hot, batch_size=batch)

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

        print("validation loss: ", loss_valid)


if __name__ == '__main__':
    train()
