import matplotlib.pyplot as plt

def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history) -> None:
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    """
    
    # plot the iterative learning curve (accuracy)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train', color='cornflowerblue')
    plt.plot(valid_loss_history, label='Valid', color='chartreuse')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(visible=True)
    plt.legend(frameon=True)

    # plot the iterative learning curve (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train', color='cornflowerblue')
    plt.plot(valid_acc_history, label='Valid', color='chartreuse')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.grid(visible=True)
    plt.legend(frameon=True)
    
    plt.show()
