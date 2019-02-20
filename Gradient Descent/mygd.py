import numpy as np


#计算总体误差
def caculator_all_loss(w,b,points):
    loss = 0
    for i in range(0,len(points)):
        x_i = points[i,0]
        y_i = points[i,1]
        loss +=((w*x_i + b)-y_i)**2
    return loss/float(2*len(points))

#根据梯度更新w和b的新值

def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (1/N)*(w_current*x + b_current - y)
        w_gradient += (1/N)*(w_current*x + b_current - y)*x
    new_b = b_current - (learningRate*b_gradient)
    new_w = w_current - (learningRate*w_gradient)
    return new_b,new_w

def gradient_descent_runner(points, starting_b, starting_w, iterations,learning_rate):
    b = starting_b
    w = starting_w
    for i in range(iterations):
        b,w = step_gradient( b, w, np.array(points),learning_rate)
    return b,w


def run():
    points = np.genfromtxt("data.csv",delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 #initial   y-intercept  guess
    initial_w = 0 #initial   slope guess
    num_iterations = 100000
    print("Staring gradient descent at b = {0},w = {1},error = {2}"
          .format(initial_b,initial_w,caculator_all_loss(initial_w,initial_b,points))
          )
    print("Running.....")
    b,w = gradient_descent_runner(points,initial_b,initial_w,num_iterations,learning_rate)
    print("After {0} iterations b = {1}, w = {2}, error = {3}"
          .format(num_iterations,b,w,caculator_all_loss(w,b,points))
          )

if __name__ == '__main__':
    run()