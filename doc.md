# 项目文档

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

请使用Visual Studio 2017（别的版本不知道是否可行）打开工程文件`CollisionDetection.sln`。

请确保您的电脑上有10.1版本的CUDA，其他版本（不要太老）应该也可以。如果环境变量（`CUDA_PATH`等）没有设置，应该需要重新配置包含目录等。

此外，`C/C++->预处理器`中的预处理器定义请加上`WIN32`（如果没有）。

不出意外的话，选择Release x64模式应该可以运行。

如果没有`nupengl.core`，可以通过`工具->NuGet包管理器->管理解决方案的NuGet程序包`来解决，在参考资料里我也附了配置`nupengl`的攻略。

### 二、文件结构

解决方案目录下的`../common, ../includes, ../packages`均为依赖，`../common`是安装CUDA时附带的Samples中的一些通用功能文件，其中`../common/inc`加入附加包含目录，`../includes`加入包含目录，`../packages`应是NuGet自动管理。

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

`DemoSystem.h/.cpp`中定义了`DemoSystem`类，其成员变量包括

- `PhysicsEngine`类实例
- `Camera`类实例
- `Shader`类实例（两个，分别用于球体和背景）

`PhysicsEngine.h/.cpp`中定义了`PhysicsEngine`类，其成员变量包括

- `SimulationEnv`类实例
- `SimulationSphereProto`类实例
  - 初始化该类需要使用`Sphere`类的数据

`PhysicsEngine`多次调用了`dSimulation.cuh/.cu`中的CUDA相关函数

`PhysicsEngine`的核心函数`update`调用了

- `hSimulation.h`中的`hSimulateFast`和`hSimulateBrutal`函数
- `dSimulation.cuh/.cu`中的`dSimulateFast`函数

`dSimulation.cu`调用了`dSimulationKernel.cuh`中的CUDA kernel函数

`dSimulationKernel.cuh`和`hSimulation.h`都调用了`mortonEncode.cuh`中的内联编码函数

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

#### 演示方法

点击`CollisionDetection.exe`程序（请确保相对目录路径不变），看到`Use default setting? Press [y] to use and any other key elsewise.`后，如果按下`y`就是使用默认设置（渲染模式，使用GPU，1000个球），然后会启动一个窗口，开始动画演示，如下图：

<img src="images\demo1.png" alt="image-20210110181814769" style="zoom:50%;" />

鼠标滚轮可以放大/缩小，键盘`WASD`可以调整视角，但相机始终面向角落（世界坐标`(0.0, 0.0, 0.0)`）。

如果之前不选择默认设置，则可以选择：

1. 是否渲染，若否则是性能测试，等待测试结果即可；
2. 是否使用GPU；
3. 球的数量，要求不大于32768（刚好初始化没有球会重合）

此外，若球数**大于4096**，只有金色小球（因为金色小球半径最小，初始化不会重合）；否则四种颜色的球都有，如上图所示。

为了使用时比较简洁，大多数参数以常量的形式写死在代码里了，可以直接在代码里修改，比如：

- `BRUTAL_MODE`，若置为`true`则使用暴力算法（CPU）
- `FRAME_RATE`，绘制的帧率，默认是50，在我的电脑上，球数超过4000左右这个帧率就有点偏高了，但只要不超过20000左右不影响观感
- `USE_SPOTLIGHT`，是否增加一个相机发出的聚光灯光源，使场景更亮
- `HORIZONTAL_FRAGMENT_NUM`和`VERTICAL_FRAGMENT_NUM`，这两个值越大一个球体的面片就越多，放大看边缘就更平滑，不过太大了会影响渲染速度

其他的没什么必要修改，与模拟环境相关的参数请不要修改。

#### 演示效果

动态效果见视频，这里放几张截图。

<img src="images\demo2.png" alt="image-20210110182859673" style="zoom:50%;" />



<img src="images\demo3.png" alt="image-20210110182956680" style="zoom:50%;" />



<img src="images\demo4.png" alt="image-20210110183115145" style="zoom:50%;" />

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

- （**主要参考**）https://www.sciencedirect.com/science/article/pii/0307904X92900352
- https://www.sciencedirect.com/science/article/abs/pii/S0301751603000322
- https://max.book118.com/html/2018/0531/169838280.shtm

Morton编码：

- https://john.cs.olemiss.edu/~rhodes/papers/Nocentino10.pdf
- https://en.wikipedia.org/wiki/Z-order_curve

Thrust文档：https://docs.nvidia.com/cuda/thrust/index.html

部分其他（问题解决）：

- CPU的按键排序实现：https://stackoverflow.com/questions/2999135/how-can-i-sort-the-vector-elements-using-members-as-the-key-in-c
- glm::rotate的使用：https://stackoverflow.com/questions/8844585/glm-rotate-usage-in-opengl

#### 引用代码出处（代码注释中也基本覆盖了）：

算法：安装CUDA后：`...\NVIDIA Corporation\CUDA Samples\v10.1\5_Simulations\particles`下的粒子模拟程序

说明：尽管我的算法思路与该程序十分相似，但有以下不同：

- DEM方法的具体形式，我查阅了一些物理文献，进行了相应的修改，并加入了球体质量和两两球体之间恢复系数的影响；
- 一些实现形式。比如球体状态的更新我并没有调用Thrust库，而是自己实现了一个CUDA kernel，更加简洁清晰；寻找cell start与end的部分没有采用复杂的块间同步和共享内存，因为该部分不是性能瓶颈，开销很小；
- 哈希函数。我使用了Morton编码，有研究证明这种编码有更好的memory access结构，从而能够提升效率；
- 其渲染与计算耦合程度更高，而我的渲染和计算完全分离；
- 总的来说，我按照自己的理解重写了整个算法，并融入了自己的理解和不同的实现。

相机类、着色器类、GLFW窗口初始化、加载纹理、绘制等：（OpenGL教程代码仓库）https://github.com/JoeyDeVries/LearnOpenGL，`shader.h`、纹理加载、窗口初始化直接借用了，着色器程序我进行了少许删改和扩展，其他均是我提炼并针对需求自己写的。

Morton编码：https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/，针对HASH最大值进行了重写。

### 六、注释与代码风格

我主要参照了Google开源风格指南：https://zh-google-styleguide.readthedocs.io/en/latest/，但也根据个人喜好调整了一些，整体风格十分统一，且注释齐全。

### 七、一点感想

我可能是为数不多的没有修上半学期的图形学而只修动画课的同学（因为上过计算机系的图形学），毫无OpenGL经验，因此大作业的开端是自学了learnopengl基础教程。之后我就开始构思算法，调研了一些八叉树实现以及NVIDIA官方例子，得出了结论：用不着八叉树，平均地划分空间并当成数组，再使用基数排序即可。NVIDIA的教程和参考资料还是相当用户友好的，帮助我快速上手了CUDA编程和空间划分算法。理解了算法+绘图的原理之后，就是慢慢地一点点写出一个完整的程序，包括渲染、相机、着色器编程、CUDA文件组织等，也在实践中发现了自己当时理解的不足。总的来说，虽然前前后后花了挺长时间，但整体的体验很棒，乐在其中。感谢这个大作业，入门OpenGL+CUDA编程+重温C++，一举三得。

### 七、附录

在这里记录一些，要求里没有提及，但我也觉得比较值得写下来的事情。