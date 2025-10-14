# Function Approximation
这个文件夹内存放的是关于函数近似的相关内容，其中 tabular_Sarsa.py、test_SARSA.py、pre-trained-SARSA.pkl 是直接使用原始的离散SARSA算法来训练mountain car (参考[github项目](https://github.com/viniciusenari/Q-Learning-and-SARSA-Mountain-Car-v0/tree/main))

而mountain_car.ipynb文件中则是我尝试使用function approximation算法解决Mountain Car问题的过程，首先尝试了线性函数近似（但这种简单的近似方法似乎无法解决Mountain Car这种非线性动力学问题），后续又使用了DQN算法来尝试解决这个问题