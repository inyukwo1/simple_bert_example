from model import Model
import torch

if __name__ == "__main__":
    # Dataset
    training_first_sentences = [
        "I am a person",
        "I like apples",
        "I have a dog",
        "I want to be a cat",
        "I am a student"
    ]

    training_second_sentences = [
        "I am a person",
        "I like pears",
        "I have a dog",
        "I want to be a cow",
        "I am a student"
    ]

    answer = torch.tensor([1., 0., 1., 0., 1.]).cuda()

    # Preparing model
    model = Model()
    model.cuda()
    bert_optimizer = torch.optim.Adam(model.bert.parameters(), lr=0.0001)
    extra_optimizer = torch.optim.Adam(model.extra_modules.parameters(), lr=0.001)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for _ in range(30):
        bert_optimizer.zero_grad()
        extra_optimizer.zero_grad()

        output = model(training_first_sentences, training_second_sentences)
        loss = loss_func(output, answer)
        loss.backward()

        print("loss: {}".format(loss.cpu().detach().numpy()))
        bert_optimizer.step()
        extra_optimizer.step()


    output = torch.nn.functional.sigmoid(model(["I like to drink"], ["I like to drink"]))
    print("Result 1: {}".format(output.cpu().detach().numpy()))
    output = torch.nn.functional.sigmoid(model(["I like to drink"], ["I want to eat pineapples"]))
    print("Result 2: {}".format(output.cpu().detach().numpy()))
    print("Result 1 should be larger than result 2")


