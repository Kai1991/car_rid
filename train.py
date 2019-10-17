from Model import CAR_RID_MODEL
from config import Config



def main():
    model = CAR_RID_MODEL(Config(),"branch_color")
    model.compile()
    model.train()

if __name__ == "__main__":
    main()