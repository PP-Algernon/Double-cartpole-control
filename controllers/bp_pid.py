
class BPNetwork:
    def __init__(self, H=10, lr=0.25, alpha=0.1):
        self.W1 = np.random.uniform(-0.5, 0.5, (H, 4))   # Hx4
        self.W2 = np.random.uniform(-0.5, 0.5, (3, H))   # 3xH
        self.lr = lr
        self.alpha = alpha
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)

    