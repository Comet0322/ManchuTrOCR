import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2)
        self.attention_weight = nn.Parameter(torch.Tensor(4, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.attention_weight, a=torch.sqrt(5))
        
    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))

        conv_outs = torch.stack([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        attention_scores = F.softmax(self.attention_weight, dim=0)
        attended_out = (conv_outs * attention_scores).sum(dim=1)
        return attended_out
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return out
    
class AMRE(nn.Module):
    def __init__(self, cnn, lstm, num_classes):
        super(AMRE, self).__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.fc = nn.Linear(lstm.hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        x = self.cnn(x)
        # Reshape x for LSTM input: (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1).contiguous()  # from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.view(x.size(0), x.size(1), -1)  # flatten the spatial dimensions
        x = self.lstm(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Hyperparameters
    input_size = 256
    hidden_size = 128
    num_layers = 2
    num_classes = 37  # Assuming 36 different glyphs + 1 blank for CTC

    cnn = AttentionCNN()
    lstm = BiLSTM(input_size, hidden_size, num_layers)
    model = AMRE(cnn, lstm, num_classes)
    input = torch.randn(2, 1, 64, 64)
    output = model(input)
    print(output.shape)