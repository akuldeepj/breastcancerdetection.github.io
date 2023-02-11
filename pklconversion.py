import pickle
from newcancerprediction import model

# save the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
