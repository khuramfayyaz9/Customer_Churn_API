import pickle

class DebugUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print("REQUESTED:", module, "->", name)
        return super().find_class(module, name)

with open("model.pkl", "rb") as f:
    obj = DebugUnpickler(f).load()

print("SUCCESS")
print(type(obj))