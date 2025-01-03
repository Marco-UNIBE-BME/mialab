import pickle

with open('inferences/pickleTest.pkl', 'wb') as f:
    pickle.dump((12,14,10000), f)

with open('inferences/pickleTest.pkl', 'rb') as f:
    bla = pickle.load(f)
    print(bla)