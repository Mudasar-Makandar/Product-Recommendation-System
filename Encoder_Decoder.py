import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class RecommenderNetwork(nn.Module):
    def __init__(self, profile_in_dim, profile_out_dim,
                 product_in_dim, hidden_dim, product_out_dim,
                 verbose=False):
        super(RecommenderNetwork, self).__init__()
        self.profile_in_dim = profile_in_dim23
        self.profile_out_dim = profile_out_dim
        self.product_in_dim = profile_in_dim
        self.hidden_dim = hidden_dim
        self.product_out_dim = product_out_dim
        self.verbose = verbose

        self.encoder = nn.GRU(product_in_dim, profile_out_dim)
        self.decoder = nn.GRU(product_out_dim, profile_out_dim)

        self.fc1 = nn.Linear(profile_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, profile_out_dim)

        self.h2o = nn.Linear(profile_out_dim, product_out_dim)


    def forward(self, profile_input, product_input,max_recommend_products = 5, device=device,ground_truth = None):

        profile = self.fc1(profile_input)
        profile = self.fc2(profile)

        #Encoder
        out, hidden = self.encoder(product_input, profile)

        if self.verbose:
            print("Encoder Product Input: ", product_input.shape)
            print("Encoder Profile Input: ", profile.shape)
            print("Encoder Output: ", out.shape)
            print("Encoder Hidden Output: ", hidden.shape)

        decoder_state = hidden
        decoder_input = torch.zeros(1, 1, self.product_out_dim).to(device)
        outputs = []

        if self.verbose:
            print('Decoder state', decoder_state.shape)
            print('Decoder input', decoder_input.shape)

        for i in range(max_recommend_products):

            out, decoder_state = self.decoder(decoder_input, decoder_state)

            if self.verbose:
                print('Decoder intermediate output', out.shape)

            out = self.h2o(decoder_state)
            outputs.append(out.view(-1,1))

            if self.verbose:
                print('Decoder output', out.shape)

            if not ground_truth is None:
                decoder_input = torch.Tensor(ground_truth[i]).view(1,1,self.product_out_dim)

            decoder_input = out
            print(decoder_input.shape)

        return outputs



if __name__=="__main__":
	net = RecommenderNetwork(profile_in_dim=17,profile_out_dim= 4,
                         product_in_dim=24, hidden_dim=32,
                         product_out_dim=24, verbose=True)
	net = net.float().to(device)
	outputs = net(profile_input, product_input, ground_truth=target)
