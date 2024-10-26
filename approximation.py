import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 0 = ô trống, 1 = X, -1 = O
        self.current_player = 1  # X đi trước

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self.get_features()

    def get_features(self):
        features = []
        # Thêm các đặc trưng như số lượng X và O trong mỗi hàng, cột, và đường chéo
        for i in range(3):
            features.append(np.sum(self.board[i, :]))  # Hàng
            features.append(np.sum(self.board[:, i]))  # Cột
        features.append(np.sum(np.diag(self.board)))  # Đường chéo
        features.append(np.sum(np.diag(np.fliplr(self.board))))  # Đường chéo ngược
        return np.array(features)

    def step(self, action):
        row, col = action // 3, action % 3
        if self.board[row, col] != 0:
            return self.get_features(), -10, True  # Phạt nếu nước đi không hợp lệ
        self.board[row, col] = self.current_player
        if self.check_win(self.current_player):
            return self.get_features(), 1, True  # Người chơi hiện tại thắng
        elif np.all(self.board != 0):
            return self.get_features(), 0, True  # Hòa
        else:
            self.current_player *= -1  # Đổi lượt chơi
            return self.get_features(), 0, False

    def check_win(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

class FunctionApproximationAgent:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.theta = np.random.randn(8)  # 8 đặc trưng
        self.alpha = alpha
        self.gamma = gamma

    def policy(self, features):
        return np.dot(self.theta, features)

    def update(self, features, reward, next_features):
        td_error = reward + self.gamma * self.policy(next_features) - self.policy(features)
        self.theta += self.alpha * td_error * features

# Huấn luyện tác nhân
agent = FunctionApproximationAgent()
env = TicTacToe()

for episode in range(10000):
    features = env.reset()
    done = False
    while not done:
        action = np.random.choice(np.where(env.board.flatten() == 0)[0])  # Chính sách ngẫu nhiên cho đơn giản
        next_features, reward, done = env.step(action)
        agent.update(features, reward, next_features)
        features = next_features
