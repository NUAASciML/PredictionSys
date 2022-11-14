import os
import pickle
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class deeponet(tf.keras.Model):
    def __init__(self):
        super(deeponet, self).__init__()
        self.trunk = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='tanh'),
                                          tf.keras.layers.Dense(64, activation='tanh'),
                                          tf.keras.layers.Dense(16)])
        self.branch_1 = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='tanh'),
                                             tf.keras.layers.Dense(64, activation='tanh'),
                                             tf.keras.layers.Dense(16)])
        self.branch_2 = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='tanh'),
                                             tf.keras.layers.Dense(64, activation='tanh'),
                                             tf.keras.layers.Dense(16)])

        self.bias = tf.Variable(0., trainable=True)

    def call(self, inputs):
        func_1 = inputs[0]
        func_2 = inputs[1]
        loc = inputs[2]

        x1 = self.branch_1(func_1)
        x2 = self.branch_2(func_2)
        x2 = x1 * x2

        x3 = self.trunk(loc)

        x4 = tf.einsum("bi,bi->b", x2, x3)
        x4 = tf.expand_dims(x4, axis=1) + self.bias

        return x4


def check_value(value):
    if value.count('.') == 1:
        left = value.split('.')[0]
        right = value.split('.')[1]
        if right.isdigit():
            if left.count('-') == 1 and left.startswith('-'):
                num = left.split('-')[-1]
                if num.isdigit():
                    return float(value)
            elif left.isdigit():
                return float(value)
        return value
    elif value.count('.') == 0:
        if value.isdigit():
            return int(value)
        elif value.count("-") == 1 and value.startswith("-"):
            num = value.split("-")[-1]
            if num.isdigit():
                return int(value)
            else:
                return value
        else:
            return value
    else:
        return value


class MyTk(tk.Tk):
    def __init__(self):
        super().__init__()
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.paned_window_all = None
        self.paned_window_part = None
        self.left_frame = None
        self.right_frame = None
        self.top_frame = None
        self.label_aoa = None
        self.value_aoa = None
        self.input_aoa = None
        self.value_epochs = None
        self.input_epochs = None
        self.aoa = None
        self.label_mach = None
        self.value_mach = None
        self.input_mach = None
        self.mach = None
        self.coordinates_path = './data/coordinates.npy'
        self.coordinates = None
        self.button_predict = None
        self.button_export = None
        self.button_exit = None
        self.max_min_path = "./data/NACA0012_DATA/train_max_min.pkl"
        self.max_min = None

        # 模型
        self.model_path_1 = "./model/deeponet_1.h5"
        self.model_path_2 = "./model/deeponet_2.h5"
        self.model_path_3 = "./model/deeponet_3.h5"
        self.init_model_1 = deeponet()
        self.init_model_1.build(input_shape=[(10000, 1), (10000, 1), (10000, 2)])
        self.init_model_2 = deeponet()
        self.init_model_2.build(input_shape=[(10000, 1), (10000, 1), (10000, 2)])
        self.init_model_3 = deeponet()
        self.init_model_3.build(input_shape=[(10000, 1), (10000, 1), (10000, 2)])

        # 结果输出
        self.sub_button1 = None
        self.sub_button2 = None
        self.sub_button3 = None
        self.result_path = "./result/"
        self.result = None
        self.result_1 = None
        self.result_2 = None
        self.result_3 = None
        self.canvas = None
        self.toolbar = None
        self.figure = None
        self.ax = None
        self.line = None
        self.first_draw = True
        self.title("流场预测系统")
        self.geometry("{}x{}+{}+{}".format(800, 500, 250, 350))
        self.resizable(False, False)
        self.create_components()

    def create_components(self):
        # 将整个界面分为左右上三块
        self.paned_window_all = tk.PanedWindow(orient=tk.HORIZONTAL, showhandle=False, sashrelief="flat")
        self.paned_window_all.pack(fill="x", expand=1)
        self.left_frame = tk.Frame(self.paned_window_all, borderwidth=2, relief="ridge")
        self.paned_window_all.add(self.left_frame, width=100, minsize=100)
        self.paned_window_part = tk.PanedWindow(orient=tk.VERTICAL, showhandle=False, sashrelief="flat")
        self.paned_window_all.add(self.paned_window_part, minsize=700)
        self.top_frame = tk.Frame(self.paned_window_part, borderwidth=1, relief="ridge")
        self.right_frame = tk.Frame(self.paned_window_part, borderwidth=0, relief="ridge")
        self.paned_window_part.add(self.top_frame, height=30, minsize=30)
        self.paned_window_part.add(self.right_frame, height=470, minsize=470)
        # 攻角
        self.label_aoa = tk.Label(self.left_frame, text="攻角")
        self.label_aoa.pack()
        self.value_aoa = tk.StringVar()
        self.input_aoa = tk.Entry(self.left_frame, borderwidth=2, relief="ridge", textvariable=self.value_aoa)
        self.input_aoa.pack()
        # 马赫数
        self.label_mach = tk.Label(self.left_frame, text="马赫数")
        self.label_mach.pack()
        self.value_mach = tk.StringVar()
        self.input_mach = tk.Entry(self.left_frame, borderwidth=2, relief="ridge", textvariable=self.value_mach)
        self.input_mach.pack()
        # 预测
        self.button_predict = tk.Button(self.left_frame, text="预测结果", width=100, command=self.predict)
        self.button_predict.pack()
        # 展示
        self.button_predict = tk.Button(self.left_frame, text="结果展示", width=100, command=self.visualize)
        self.button_predict.pack()
        # 导出结果
        self.button_export = tk.Button(self.left_frame, text="导出结果", width=100, command=self.export)
        self.button_export.pack()
        # 退出
        self.button_exit = tk.Button(self.left_frame, text="退出", width=100, command=self.exit)
        self.button_exit.pack()

    def predict(self):
        # 先判断攻角是否合法
        temp_value_aoa = self.value_aoa.get()
        checked_aoa = check_value(temp_value_aoa)
        if isinstance(checked_aoa, float) or isinstance(checked_aoa, int):
            if checked_aoa < -10 or checked_aoa > 10:
                messagebox.showwarning("警告", "攻角范围为-10~10")
            else:
                temp_value_mach = self.value_mach.get()
                checked_mach = check_value(temp_value_mach)
                # 攻角合法之后判断马赫数是否合法
                if isinstance(checked_mach, float) or isinstance(checked_mach, int):
                    if checked_mach > 1 or checked_mach < 0:
                        messagebox.showwarning("警告", "马赫数范围为0~1")
                    else:
                        # 查看坐标文件是否存在
                        if os.path.exists(self.coordinates_path):
                            self.coordinates = np.load(self.coordinates_path)
                            zeros = np.zeros((self.coordinates.shape[0], 1))
                            self.aoa = checked_aoa
                            self.mach = checked_mach

                            x = self.coordinates[:, 0]
                            y = self.coordinates[:, 1]
                            index_x2 = (0 <= x) & (x <= 1.001)
                            index_y2 = (-0.06 <= y) & (y <= 0)
                            index_all2 = index_x2 & index_y2
                            temp_x2 = x[index_all2][:83]
                            temp_y2 = y[index_all2][:83]
                            x_id2 = temp_x2.argsort(axis=0)
                            self.x2 = temp_x2[x_id2]
                            self.y2 = temp_y2[x_id2]
                            self.x1 = self.x2
                            self.y1 = -self.y2

                            checked_aoa += tf.constant(zeros, dtype=tf.float32)
                            checked_mach += tf.constant(zeros, dtype=tf.float32)
                            # 查看model文件夹下是否有模型文件
                            if os.path.exists(self.model_path_1) and os.path.exists(self.model_path_2) and os.path.exists(
                                    self.model_path_3):
                                # 在此计算结果并更新result以便导出数据
                                self.init_model_1.load_weights(self.model_path_1)
                                self.init_model_2.load_weights(self.model_path_2)
                                self.init_model_3.load_weights(self.model_path_3)

                                self.result_1 = self.init_model_1.call([checked_aoa, checked_mach, self.coordinates])
                                self.result_2 = self.init_model_2.call([checked_aoa, checked_mach, self.coordinates])
                                self.result_3 = self.init_model_3.call([checked_aoa, checked_mach, self.coordinates])

                                self.result = tf.concat([self.result_1, self.result_2, self.result_3], -1).numpy()
                                self.result_1 = self.result_1.numpy().T[0]
                                self.result_2 = self.result_2.numpy().T[0]
                                self.result_3 = self.result_3.numpy().T[0]

                                if os.path.exists(self.max_min_path):
                                    self.max_min = np.array(pickle.load(open(self.max_min_path, "rb"))).T
                                    self.result = self.result.T
                                    for i in range(3):
                                        self.result[i] = self.result[i] * (self.max_min[i, 0] - self.max_min[i, 1]) + self.max_min[i, 1]
                                    self.result = self.result.T
                                    messagebox.showinfo("提示", "结果预测成功")
                                else:
                                    messagebox.showerror("错误", "归一化文件不存在")
                            else:
                                messagebox.showerror("错误", "模型文件不存在")
                        else:
                            messagebox.showerror("错误", "坐标文件不存在")
                else:
                    messagebox.showwarning("警告", "马赫数必须是数字类型")
        else:
            messagebox.showwarning("警告", "攻角必须是数字类型")

    def export(self):
        # 看是否已经有预测结果
        if self.result is None:
            messagebox.showinfo("提示", "请先预测结果再导出")
        else:
            is_exist = os.path.isdir(self.result_path)
            if not is_exist:
                # 防止某个文件名叫result
                if os.path.isfile("./result"):
                    os.remove("./result")
                os.makedirs(self.result_path)
            # 用攻角、马赫数、时间给结果命名
            now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S").replace('-', '')
            output = open(self.result_path + f"aoa{self.aoa}_mach{self.mach}_result_{now_time}.pkl", 'wb')
            pickle.dump(self.result, output)
            output.close()
            messagebox.showinfo("提示", "结果导出成功")

    def exit(self):
        self.destroy()

    def visualize(self, num=1):
        if self.result is None:
            messagebox.showinfo("提示", "请先预测结果再展示")
        else:
            # 第一次画图的画需要新建画布
            if self.first_draw:
                self.sub_button1 = tk.Button(self.top_frame, text="x分量速度", width=20, command=lambda: self.visualize(1))
                self.sub_button1.pack(side='left')
                self.sub_button2 = tk.Button(self.top_frame, text="y分量速度", width=20, command=lambda: self.visualize(2))
                self.sub_button2.pack(side='left')
                self.sub_button3 = tk.Button(self.top_frame, text="压力值", width=20, command=lambda: self.visualize(4))
                self.sub_button3.pack(side='left')

                self.figure = Figure(figsize=(7, 4.2), dpi=100)
                self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
                x = self.coordinates[:, 0]
                y = self.coordinates[:, 1]

                self.ax = self.figure.add_subplot()
                if num == 1:
                    cax = self.ax.tripcolor(x, y, self.result_1, label='x_speed', shading='gouraud', cmap="GnBu")
                    self.ax.set_title("x_speed")
                elif num == 2:
                    cax = self.ax.tripcolor(x, y, self.result_2, label='y_speed', shading='gouraud', cmap="OrRd")
                    self.ax.set_title("y_speed")
                else:
                    cax = self.ax.tripcolor(x, y, self.result_3, label='pressure', shading='gouraud', cmap="rainbow")
                    self.ax.set_title("pressure")
                self.ax.fill_between(self.x1, self.y1, facecolor='white')
                self.ax.fill_between(self.x2, self.y2, facecolor='white')

                self.ax.set_xlim(-0.5, 1.5)
                self.ax.set_ylim(-1.0, 1.0)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.figure.colorbar(cax)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack()
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame, pack_toolbar=False)
                self.toolbar.update()
                self.toolbar.pack()
                self.first_draw = False

            else:
                self.figure.clear()
                x = self.coordinates[:, 0]
                y = self.coordinates[:, 1]
                self.ax = self.figure.add_subplot()
                if num == 1:
                    cax = self.ax.tripcolor(x, y, self.result_1, label='x_speed', shading='gouraud', cmap="GnBu")
                    self.ax.set_title("x_speed")
                elif num == 2:
                    cax = self.ax.tripcolor(x, y, self.result_2, label='y_speed', shading='gouraud', cmap="OrRd")
                    self.ax.set_title("y_speed")
                else:
                    cax = self.ax.tripcolor(x, y, self.result_3, label='pressure', shading='gouraud', cmap="rainbow")
                    self.ax.set_title("pressure")
                self.ax.fill_between(self.x1, self.y1, facecolor='white')
                self.ax.fill_between(self.x2, self.y2, facecolor='white')

                self.ax.set_xlim(-0.5, 1.5)
                self.ax.set_ylim(-1.0, 1.0)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.figure.colorbar(cax)
                self.canvas.draw()


if __name__ == "__main__":
    root = MyTk()
    root.mainloop()
