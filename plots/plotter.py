import matplotlib.pyplot as plt

titles = ["SQA", "SQA + COMET", "SQA + SQUAD", "SQA + SQUAD + COMET"]
types = ["Train Loss", "Eval Loss", "Eval Accs"]

eval_accs = []
train_losses = []
eval_losses = []

def reader(fname1, fname2):
    
    f1, f2 = open(fname1), open(fname2)
    a, b, c = [], [], []
    
    for line in f1:
        vals = line.split(',')
        vals = [float(val) for val in vals]
        c.append(vals[0])
            
    for line in f2:
        vals = line.split(',')
        vals = [float(val) for val in vals]
        a.append(vals[0])
        b.append(vals[1])
        
    return a, b, c

a_1, b_1, c_1 = reader("sqa_train_loss_logs.txt", "sqa_train_eval_logs.txt")
a_2, b_2, c_2 = reader("comet_train_loss_logs.txt", "comet_train_eval_logs.txt")
a_3, b_3, c_3 = reader("comet_train_loss_logs_squad_sqa.txt", "comet_train_eval_logs_squad_sqa.txt")
              
a_4_tmp = open("asa_eval_scores.txt")
b_4_tmp = open("asa_eval_losses.txt")
c_4_tmp = open("asa_train_losses.txt")

a_4 = [float(v) for v in a_4_tmp.readlines()]
b_4 = [float(v) for v in b_4_tmp.readlines()]
c_4 = [float(v) for v in c_4_tmp.readlines()]

lenx = min(len(a_1), len(a_2), len(a_3), len(a_4))
x_vals = [i*500 for i in range(lenx)]

plt.plot(x_vals, a_1[:lenx], label = titles[0] + " " + types[2])
plt.plot(x_vals, a_2[:lenx], label = titles[1] + " " + types[2])
plt.plot(x_vals, a_3[:lenx], label = titles[2] + " " + types[2])
plt.plot(x_vals, a_4[:lenx], label = titles[3] + " " + types[2])
plt.title(types[2])
plt.legend()
plt.show()

plt.plot(x_vals, b_1[:lenx], label = titles[0] + " " + types[1])
plt.plot(x_vals, b_2[:lenx], label = titles[1] + " " + types[1])
plt.plot(x_vals, b_3[:lenx], label = titles[2] + " " + types[1])
plt.plot(x_vals, b_4[:lenx], label = titles[3] + " " + types[1])
plt.title(types[1])
plt.legend()
plt.show()

plt.plot(x_vals, c_1[:lenx], label = titles[0] + " " + types[0])
plt.plot(x_vals, c_2[:lenx], label = titles[1] + " " + types[0])
plt.plot(x_vals, c_3[:lenx], label = titles[2] + " " + types[0])
plt.plot(x_vals, c_4[:lenx], label = titles[3] + " " + types[0])
plt.title(types[0])
plt.legend()
plt.show()