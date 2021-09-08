def standard(epochs, network, loader_function, optimizer, criterion,
             print_every=100, accuracy_function=None):
    # TODO!! save best model after a given number of iterations
    for i in range(epochs):
        running_loss = total_items = running_acc = 0
        for x, y in loader_function():
            items_count = len(y)
            optimizer.zero_grad()

            y_hat = network(x)
            loss = criterion(y_hat, y)
            running_loss += loss.item() * items_count
            total_items += items_count
            if accuracy_function:
                running_acc += accuracy_function(y_hat, y)

            loss.backward()
            optimizer.step()
        if i % print_every == 0:
            print("Iter {0} Avg. Loss: {1}".format(i, running_loss / total_items))
            if accuracy_function:
                print("Iter {0} Avg. Accuracy: {1}".format(i, running_acc / total_items))
