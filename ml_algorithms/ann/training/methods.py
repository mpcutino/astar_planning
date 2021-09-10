def standard(epochs, network, loader_function, optimizer, criterion,
             print_every=100, accuracy_function=None):
    # TODO!! save best model after a given number of iterations. Add Early Stop
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
        if i % print_every == 0 or i == epochs - 1:
            print("==== Iter {0} =====".format(i))
            print("Avg. Loss: {0}".format(running_loss / total_items))
            if accuracy_function:
                print("Avg. Accuracy: {0}".format(running_acc / total_items))
