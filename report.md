# 文档

2017013599 软件71 从业臻

### 一、运行环境

操作系统：Windows 10

IDE：Visual Studio 2017

CUDA版本：10.1

依赖的库与文件包括：

- glad和GLFW（通过nupengl.core和nupengl.core.redist）
- stb_image.h（用于纹理加载）
- glm数学运算库
- NVIDIA GPU Computing Toolkit
- CUDA samples里的一些文件（`...\CUDA Samples\v10.1\common`）

#### 环境配置

使用Visual Studio 2017打开工程文件`CollisionDetection.sln`。

请确保您的电脑上有10.1版本的CUDA，其他版本（不要太老）应该也可以。

此外，`C/C++->预处理器`中的预处理器定义请加上`WIN32`（如果没有）。

不出意外的话，选择Release x64模式应该可以运行。

如果没有`nupengl.core`，可以通过`工具->NuGet包管理器->管理解决方案的NuGet程序包`来解决。

### 二、文件结构

解决方案目录下的`../common,../includes,../packages`均为依赖，`../common`是安装CUDA时附带的Samples中的一些通用功能文件，其中`../common/inc`加入附加包含目录，`../includes`加入包含目录，`../packages`应是NuGet自动管理。

`../shaderPrograms`中是几个着色器程序，都是基于learnopengl教程网站提供的样例改写；`../resources`则是learnopengl教程网站提供的资源中选取的几张纹理。

进入项目目录：

- `main.cpp`是入口点
- `glad.c`是为了使glad正常工作必须包含的源文件；
- `camera.h`是相机类
- `shader.h`是着色器类，主要用于加载并绑定着色器程序等一系列操作
- `global.h`里定义了一些常量和一些可以更改的变量
- `environment.h`里定义了模拟场景时的诸多参数
- `sphere.h`里定义了几种球的原型，和辅助的类
- `demoSystem.h/.cpp`里是演示系统（包含渲染系统和单纯性能测试）的实现
- `physicsEngine.h/.cpp`里是物理引擎（即场景构建，碰撞检测模拟）的实现
- `hSimulation.h`里是CPU（host）上的碰撞检测算法（包括基于空间划分和暴力）的实现
- `dSimulation.cuh/.cu`里是GPU（device）上的碰撞检测算法（基于空间划分）的实现，但不包括kernel代码
- `dSimulationKernel.cu`是碰撞检测算法的CUDA kernel代码
- `mortonEncode.cuh`里包括了CPU/GPU上的Morton编码实现（用于哈希）
- `packages.config`等，Visual Studio配置相关文件

### 三、程序逻辑

在上述文件用途简述的基础上，详细阐释程序模块间的逻辑和主要运行流程。

#### 模块之间的逻辑



#### 主要运行流程

`main.cpp`构造一个`DemoSystem`实例`demo_system`，根据`RENDER_MODE`的值，选择：

- 进入演示动画
- 开始性能测试

对于前者，`demo_system`会初始化`GLFW`窗口，加载纹理、着色器，初始化数据、顶点数组、缓冲等，即准备好渲染部分；然后`demo_system`的成员`physics_engine_`也会初始化，分配好内存、显存，初始化一些必要数据。

之后进入一个循环，主要有这几件事：

- 处理`GLFW`窗口事件，检测输入等
- `physics_engine_`执行`update`函数，获取所有球体最新的位置，并返回给`demo_system`（如果是GPU模式则要从显存复制到内存）
- 绘制背景和所有球体
- 睡眠（使得帧数基本恒定）

对于后者，区别是不会涉及到任何的`GLFW`窗口初始化和任何绘制相关的逻辑。程序静默地运行一定次数的`physics_engine_.update`，然后输出消耗的时间。

### 四、功能演示



### 五、参考文献和引用代码出处

#### 参考文献（包括网络资源）：

CUDA安装：https://www.cnblogs.com/arxive/p/11198420.html

OpenGL相关库和包安装：http://fangda.me/2018/03/17/VS2017-%E9%85%8D%E7%BD%AEglut-glew-glfw-glad%E6%9C%80%E7%AE%80%E5%8D%95%E7%9A%84%E6%96%B9%E6%B3%95/

CUDA使用：

- https://docs.nvidia.com/cuda/

- https://developer.nvidia.com/zh-cn/blog

OpenGL使用：（**主要参考**）https://learnopengl-cn.github.io/

空间划分算法：

- （**主要参考**）https://developer.download.nvidia.com/assets/cuda/files/particles.pdf
- https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda

球体碰撞后状态更新方式（DEM方法）:

- https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus

DEM方法论文：

- https://max.book118.com/html/2018/0531/169838280.shtm

Morton编码：

- https://john.cs.olemiss.edu/~rhodes/papers/Nocentino10.pdf
- https://en.wikipedia.org/wiki/Z-order_curve

代码风格（Google）：https://zh-google-styleguide.readthedocs.io/en/latest/

Thrust文档：https://docs.nvidia.com/cuda/thrust/index.html

部分其他（问题解决）：

- CPU的按键排序实现：https://stackoverflow.com/questions/2999135/how-can-i-sort-the-vector-elements-using-members-as-the-key-in-c
- glm::rotate的使用：https://stackoverflow.com/questions/8844585/glm-rotate-usage-in-opengl

#### 引用代码出处（代码注释中也基本覆盖了）：

算法：安装CUDA后：`...\NVIDIA Corporation\CUDA Samples\v10.1\5_Simulations\particles`下的粒子模拟程序

相机类、着色器类、GLFW窗口初始化、加载纹理、绘制等：（OpenGL教程代码仓库）https://github.com/JoeyDeVries/LearnOpenGL

Morton编码：https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/