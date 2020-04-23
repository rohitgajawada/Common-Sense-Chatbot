import matplotlib.pyplot as plt

x = []
y = []
z = []

# f = open("sqa_train_loss_logs.txt")
# f = open("sqa_train_eval_logs.txt")
# f = open("comet_train_loss_logs.txt")
# f = open("comet_train_eval_logs.txt")
# f = open("comet_train_loss_logs_squad_sqa.txt")
f = open("comet_train_eval_logs_squad_sqa.txt")

for line in f:
    vals = line.split(',')
    # print(line)
    vals = [float(val) for val in vals]
    if len(vals) == 3:
        x.append(vals[2])
        y.append(vals[0])
        z.append(vals[1])
    else:
        x.append(vals[1])
        z.append(vals[0])
        
        
if len(y) == 0:
    
    plt.plot(x, z)
    plt.title("COMET SQA SQUAD Train Loss")
    plt.show()
    
else:
    plt.plot(x, z)
    plt.title("COMET SQA SQUAD Eval Loss")
    plt.show()
    
    plt.plot(x, y)
    plt.title("COMET SQA SQUAD Eval Acc")
    plt.show()