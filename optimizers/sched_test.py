import torch
from lr_scheduler import LinearWarmupCosineAnnealingLR
torch.random.manual_seed(0)


def start_train():
    inp = torch.ones(10)

    net = torch.nn.Linear(10, 10)
    optim = torch.optim.AdamW(net.parameters(), lr=0.001)
    sched = LinearWarmupCosineAnnealingLR(optim, warmup_epochs=5, max_epochs=200)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        # print(sched.get_last_lr())

        out = net(inp)
        loss = loss_func(out, inp)
        loss.backward()
        optim.step()
        sched.step()

        print(loss.item())

    torch.save({
        "scheduler": sched.state_dict(),
        "optimizer": optim.state_dict(),
        "state_dict": net.state_dict()
    }, "weights.pt")


def resume_train():
    inp = torch.ones(10)

    weights = torch.load("weights.pt")

    net = torch.nn.Linear(10, 10)
    optim = torch.optim.AdamW(net.parameters(), lr=0.001)

    sched = LinearWarmupCosineAnnealingLR(optim, warmup_epochs=5, max_epochs=100)
    loss_func = torch.nn.MSELoss()

    net.load_state_dict(weights["state_dict"])
    optim.load_state_dict(weights["optimizer"])
    sched.load_state_dict(weights["scheduler"])

    for i in range(100):
        # print(sched.get_last_lr())

        out = net(inp)
        loss = loss_func(out, inp)
        loss.backward()
        optim.step()
        sched.step()

        print(loss.item())



if __name__ == '__main__':
    start_train()
    print("=======SWITCH=======")
    resume_train()