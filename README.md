[toc]

# PredictionSys

在这个仓库中，我们上传了预测系统的源代码和预训练好的AI模型。除此之外，我们会给出一些指导，以便用户更好地使用我们的软件。

## Python环境安装

首先，打开命令行，然后输入 `python`。如果你已经成功安装了`python`，你将看到与它的版本相关的信息和其他内容。否则，你需要安装`python`。

## 依赖包

确保`Python`正确安装后，你需要安装本项目所需要的依赖包。

请在本项目根目录下打开命令行，并且在命令行中输入如下命令：

```shell
pip install -r requirements.txt
```

推荐国内用户使用镜像加快安装速度：

```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 使用说明

1. 安装完项目所使用的依赖之后，你可以开始使用本系统，请在项目根目录下打开命令行并输入以下命令以启动主界面：

   ```shell
   python main.py
   ```

2. 进入主界面以后，输入攻角和马赫数，点击“`预测结果`”按钮，系统开始对流场物理量进行预测。

3. 待系统提示“`结果预测成功后`”，点击“`结果展示`”按钮，进行预测结果的可视化。

4. 点击“`导出结果`”按钮，即可将预测成功的流场物理量进行导出。（此步骤需在结果预测成功之后方可有效）

5. 点击“`退出`”按钮，即可退出本系统。

## 模型的重新训练

打开项目根目录下的`deeponet.ipynb`文件（推荐使用`jupyter notebook`打开），设置想要训练的轮数`num_epochs`和批大小`batchsize`，然后运行即可。

## Tips

1. 软件依赖安装可能会耗时较长，请耐心等待。
2. 本项目只针对华为赛题二中给定的`NACA0012`翼型的流场数据进行AI模型的构建，如需扩展至其他翼型，可在后续开发过程中开发类似功能。
3. 删除项目中已存在的某些文件可能会导致项目报错，请谨慎删除。
4. 重新训练得到的新模型参数会覆盖原有的模型参数。
