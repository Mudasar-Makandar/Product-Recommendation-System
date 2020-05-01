from DataLoader import DataReader
from Encoder_Decoder import RecommenderNetwork

#====test======
RecommenderNetwork(profile_in_dim=17,profile_out_dim= 4,
                     product_in_dim=24, hidden_dim=32,
                     product_out_dim=24, verbose=True)
